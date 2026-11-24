import h5py
import numpy as np
import matplotlib.pyplot as plt
file_path = "C:\\Users\\vishw\\ml tut\\Project\\data\\MICRO2D_homogenized.h5"  # adjust if your filename is different
CLASS = "GRF"
with h5py.File(file_path, "r") as f:
    images = f[f"{CLASS}/{CLASS}"][:]
    mech = f[f"{CLASS}/homogenized_mechanical"][:]
E_x = mech[:, 0, 0]
print(f"Images shape:   {images.shape}")
print(f"Property shape: {E_x.shape}")
print(f"E_x range:      {E_x.min():.4f} to {E_x.max():.4f}")
print(f"E_x mean:       {E_x.mean():.4f}")
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
indices = np.random.randint(0, len(images), 6)
for ax, idx in zip(axes.flat, indices):
    ax.imshow(images[idx], cmap="gray")
    ax.set_title(f"E_x = {E_x[idx]:.4f}", fontsize=9)
    ax.axis("off")

plt.suptitle("6 random GRF microstructures with E_x labels", fontsize=11)
plt.tight_layout()
plt.savefig("C:\\Users\\vishw\\ml tut\\Project\\outputs\\sanity_check.png", dpi=120)
plt.show()
