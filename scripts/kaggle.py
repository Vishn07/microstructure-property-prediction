import kagglehub
import os

# Download dataset (or get cached path if already downloaded)
path = kagglehub.dataset_download("safi842/highcarbon-micrographs")

print("Dataset path:", path)

# List files
for root, dirs, files in os.walk(path):
    for file in files[:5]:
        print(os.path.join(root, file))