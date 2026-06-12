# Microstructure → Property Prediction

ML pipeline predicting elastic modulus (E_x) from two-phase
microstructure images. Random Forest on 19 hand-crafted features
achieves R² = 0.91 on a held-out test set of the MICRO2D dataset.

## What it does
- Extracts 19 image features per micrograph (GLCM texture, FFT
  energy, phase fraction, edge density, grain-boundary metrics)
- Trains and validates a Random Forest regressor (10,000 images)
- Single-image inference: drop in a new micrograph, get a
  property prediction

## Results
| Model | R² (test) |
|-------|-----------|
| Random Forest, 19 features | 0.91 |

## How to run
pip install -r requirements.txt
python train_model.py        # expects MICRO2D HDF5 (see below)
python inference.py path/to/image.png

## Data
Uses the MICRO2D synthetic dataset [link]. Not included in repo —
download instructions here.

## Context
Built in collaboration with [lab context], working toward
predicting yield/tensile strength from real steel micrographs.
