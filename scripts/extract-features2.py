import h5py
import numpy as np
import pandas as pd
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
FILE_PATH ="C:\\Users\\vishw\\ml tut\\Project\\data\\MICRO2D_homogenized.h5"
OUTPUT_CSV="C:\\Users\\vishw\\ml tut\\Project\\outputs\\features_v2.csv"
CLASS = "GRF"
N_SAMPLES = 10000
with h5py.File(FILE_PATH, "r") as f:
    images = f[f"{CLASS}/{CLASS}"][:N_SAMPLES]
    mech   = f[f"{CLASS}/homogenized_mechanical"][:N_SAMPLES]
E_x = mech[:, 0, 0]
def mean_intensity(img):
    return img.mean()

def edge_density(img):
    img_uint8 = (img * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, threshold1=100, threshold2=200)
    return edges.mean() / 255.0

def texture_variance(img):
    return float(img.var() * 255 * 255)

def glcm_features(img, distances=[1]):
    """GLCM at multiple distances — captures structure at different scales."""
    img_uint8 = (img * 255).astype(np.uint8)
    img_64 = (img_uint8 // 4).astype(np.uint8)
    results = {}
    for d in distances:
        gcm = graycomatrix(img_64, distances=[d], angles=[0],
                           levels=64, symmetric=True, normed=True)
        results[f"glcm_contrast_d{d}"]    = graycoprops(gcm, "contrast")[0, 0]
        results[f"glcm_homogeneity_d{d}"] = graycoprops(gcm, "homogeneity")[0, 0]
        results[f"glcm_energy_d{d}"]      = graycoprops(gcm, "energy")[0, 0]
        results[f"glcm_correlation_d{d}"] = graycoprops(gcm, "correlation")[0, 0]
    return results

def phase_fraction(img):
    """Fraction of pixels belonging to phase 1 """
    return float((img == 1).sum()) / img.size

def fft_energy_ratio(img):
    """
    Ratio of low-frequency to total FFT energy.
    High ratio = large scale ordered structure.
    Low ratio  = fine chaotic structure.
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2

    # define low frequency region as central 10% of the spectrum
    h, w = magnitude.shape
    cx, cy = h // 2, w // 2
    radius = min(h, w) // 10
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    low_freq_energy  = magnitude[mask].sum()
    total_energy     = magnitude.sum()
    return float(low_freq_energy / (total_energy + 1e-10))

def run_length_mean(img):
    """
    Average length of consecutive same-phase runs across all rows.
    Long runs = coarse large phases.
    Short runs = fine dispersed phases.
    """
    run_lengths = []
    for row in img:
        if len(row) == 0:
            continue
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                run_lengths.append(count)
                count = 1
        run_lengths.append(count)
    return float(np.mean(run_lengths)) if run_lengths else 0.0

rows = []
DISTANCES = [1, 2, 4, 8]

for i in tqdm(range(len(images))):
    img = images[i].astype(np.float64)

    row = {"index": i}

    # original features
    row["mean_intensity"]   = round(mean_intensity(img), 6)
    row["edge_density"]     = round(edge_density(img), 6)
    row["texture_variance"] = round(texture_variance(img), 4)

    # glcm at multiple distances
    glcm = glcm_features(img, distances=DISTANCES)
    for k, v in glcm.items():
        row[k] = round(v, 6)

    # new features
    row["phase_fraction"]   = round(phase_fraction(img), 6)
    row["fft_energy_ratio"] = round(fft_energy_ratio(img), 6)
    row["run_length_mean"]  = round(run_length_mean(img), 4)

    row["E_x"] = round(E_x[i], 4)
    rows.append(row)
    os.makedirs("C:\\Users\\vishw\\ml tut\\Project\\outputs", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Done")
print(f"Features per sample: {len(df.columns) - 2}")  # exclude index and E_x
print("\nFeature columns:")
print([c for c in df.columns if c not in ["index", "E_x"]])
print("\nFirst 3 rows:")
print(df.head(3).to_string())