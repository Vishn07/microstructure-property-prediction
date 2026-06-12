import h5py
import numpy as np
import pandas as pd
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
FILE_PATH ="C:\\Users\\vishw\\ml tut\\Project\\data\\MICRO2D_homogenized.h5"
OUTPUT_CSV="C:\\Users\\vishw\\ml tut\\Project\\outputs\\features.csv"
CLASS ="GRF"
N_SAMPLES=10000
with h5py.File(FILE_PATH, "r") as f:
    images = f[f"{CLASS}/{CLASS}"][:N_SAMPLES]
    mech   = f[f"{CLASS}/homogenized_mechanical"][:N_SAMPLES]
E_x = mech[:, 0, 0]
print(f"Loaded {len(images)} images")
def mean_intensity(img):
    """Average pixel value — proxy for phase fraction."""
    return img.mean()
def edge_density(img):
    """Fraction of pixels that are edges — proxy for grain boundary density."""
    img_uint8 = (img * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, threshold1=100, threshold2=200)
    return edges.mean() / 255.0
def texture_variance(img):
    """Local variance of pixel values — captures microstructure roughness."""
    return float(img.var() * 255 * 255)
def glcm_features(img):
    """
    Gray-Level Co-occurrence Matrix features.
    Captures spatial relationships between pixels — very powerful for microstructures.
    Returns: contrast, homogeneity, energy, correlation
    """
    img_uint8 = (img * 255).astype(np.uint8)
    # Rescale to 64 grey levels to keep computation fast
    img_64 = (img_uint8 // 4).astype(np.uint8)
    gcm = graycomatrix(img_64, distances=[1], angles=[0],levels=64, symmetric=True, normed=True)
    contrast    = graycoprops(gcm, "contrast")[0, 0]
    homogeneity = graycoprops(gcm, "homogeneity")[0, 0]
    energy      = graycoprops(gcm, "energy")[0, 0]
    correlation = graycoprops(gcm, "correlation")[0, 0]
    return contrast, homogeneity, energy, correlation
sample = images[0]
print(f"dtype:        {sample.dtype}")
print(f"min value:    {sample.min()}")
print(f"max value:    {sample.max()}")
print(f"unique vals:  {np.unique(sample)}")
rows = []

for i in tqdm(range(len(images))):
    img = images[i]   # shape (256, 256), dtype int64

    mi  = mean_intensity(img)
    ed  = edge_density(img)
    tv  = texture_variance(img)
    c, h, en, co = glcm_features(img)

    rows.append({
        "index":        i,
        "mean_intensity":   round(mi, 6),
        "edge_density":     round(ed, 6),
        "texture_variance": round(tv, 4),
        "glcm_contrast":    round(c,  6),
        "glcm_homogeneity": round(h,  6),
        "glcm_energy":      round(en, 6),
        "glcm_correlation": round(co, 6),
        "E_x":              round(E_x[i], 4),
    })
os.makedirs("C:\\Users\\vishw\\ml tut\\Project\\outputs", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. {len(df)} rows saved to {OUTPUT_CSV}")
print("\nFirst 3 rows:")
print(df.head(3).to_string())
print("\nFeature stats:")
print(df.drop(columns=["index"]).describe().round(4).to_string())