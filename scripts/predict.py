import numpy as np
import pandas as pd
import cv2
import pickle
import os
import sys
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
MODEL_PATH = "C:\\Users\\vishw\\ml tut\\Project\\outputs\\rf_model.pkl"
def mean_intensity(img):
    return img.mean()

def edge_density(img):
    img_uint8 = (img * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, threshold1=100, threshold2=200)
    return edges.mean() / 255.0

def texture_variance(img):
    return float(img.var() * 255 * 255)

def glcm_features(img, distances=[1]):
    img_uint8 = (img * 255).astype(np.uint8)
    img_64 = (img_uint8 // 4).astype(np.uint8)
    results = {}
    gcm = graycomatrix(img_64, distances=[1], angles=[0],
                       levels=64, symmetric=True, normed=True)
    results["glcm_contrast"]    = graycoprops(gcm, "contrast")[0, 0]
    results["glcm_homogeneity"] = graycoprops(gcm, "homogeneity")[0, 0]
    results["glcm_energy"]      = graycoprops(gcm, "energy")[0, 0]
    results["glcm_correlation"] = graycoprops(gcm, "correlation")[0, 0]
    return results

def phase_fraction(img):
    return float((img > 0.5).sum()) / img.size

def fft_energy_ratio(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2
    h, w = magnitude.shape
    cx, cy = h // 2, w // 2
    radius = min(h, w) // 10
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    low_freq_energy = magnitude[mask].sum()
    total_energy = magnitude.sum()
    return float(low_freq_energy / (total_energy + 1e-10))

def run_length_mean(img):
    binary = (img > 0.5).astype(int)
    run_lengths = []
    for row in binary:
        count = 1
        for j in range(1, len(row)):
            if row[j] == row[j-1]:
                count += 1
            else:
                run_lengths.append(count)
                count = 1
        run_lengths.append(count)
    return float(np.mean(run_lengths)) if run_lengths else 0.0

def extract_features(img_normalized):
    """Takes a normalized 0-1 float image, returns feature dict."""
    row = {}
    row["mean_intensity"]   = mean_intensity(img_normalized)
    row["edge_density"]     = edge_density(img_normalized)
    row["texture_variance"] = texture_variance(img_normalized)

    glcm = glcm_features(img_normalized)
    for k, v in glcm.items():
        row[k] = v

    row["phase_fraction"]   = phase_fraction(img_normalized)
    row["fft_energy_ratio"] = fft_energy_ratio(img_normalized)
    row["run_length_mean"]  = run_length_mean(img_normalized)
    return row
def load_image(image_path):
    """
    normalize images
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    # resize to 256x256 to match training data
    if img.shape != (256, 256):
        print(f"  Resizing from {img.shape} to (256, 256)")
        img = cv2.resize(img, (256, 256))

    # normalize to 0-1
    img_normalized = img.astype(np.float64) / 255.0
    return img_normalized

def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print("ERROR: No saved model found.")
        return

    # load model
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    rf       = model_data["model"]
    features = model_data["feature_names"]

    print(f"\nModel loaded. Expects {len(features)} features:")
    print(features)

    # load and process image
    print(f"\nLoading image: {image_path}")
    img = load_image(image_path)
    print(f"Image loaded. Shape: {img.shape}, "
          f"value range: {img.min():.3f} to {img.max():.3f}")
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.title("Input image (normalized)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("C:\\Users\\vishw\\ml tut\\Project\\outputs\\input_image.png", dpi=120)
    plt.show()
    print("Image saved to outputs/input_image.png")
    # extract features
    print("\nExtracting features...")
    feat_dict = extract_features(img)
    print("\nAvailable feature keys:")
    print(list(feat_dict.keys()))
    print("\nModel expects:")
    print(features)
    # build feature vector in correct order
    X = np.array([[feat_dict[f] for f in features]])

    # print features so you can see what was extracted
    print("\nExtracted features:")
    for name, val in zip(features, X[0]):
        print(f"  {name:<25} {val:.6f}")

    # predict
    prediction = rf.predict(X)[0]
    print(f"\n── Prediction ──────────────────────")
    print(f"Predicted E_x: {prediction:.2f}")
    print(f"(Training data range: 1015 to 4981)")
    return prediction
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/your/image.png")
    else:
        image_path = sys.argv[1]
        prediction = predict(image_path)

        # parse true value from filename
        # expects format: GRF_idx7453_Ex1242.png
        filename = os.path.basename(image_path)
        try:
            true_val = float(filename.split("_Ex")[1].replace(".png", ""))
            error    = abs(prediction - true_val)
            pct_err  = (error / true_val) * 100
            print(f"\n── Comparison ──────────────────────")
            print(f"True E_x:      {true_val:.2f}")
            print(f"Predicted E_x: {prediction:.2f}")
            print(f"Error:         {error:.2f}  ({pct_err:.1f}%)")
        except:
            print("Could not parse true value from filename")