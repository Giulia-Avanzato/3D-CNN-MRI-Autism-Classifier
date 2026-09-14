
import os
import csv
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# Valori di crop trovati per tentativi, tali da rimuovere solo le porzioni
# di spazio vuoto (non informative) attorno al cervello.
# Applicati sui tre assi dell'immagine originale (61 x 73 x 61).
CROP_X = (7, 54)   # asse 0
CROP_Y = (5, 68)   # asse 1
CROP_Z = (6, 55)   # asse 2


def read_nii(filepath, crop=True):
    img = nib.load(filepath)
    data = img.get_fdata()  # array numpy grezzo, shape (61, 73, 61)

    if crop:
        data = data[CROP_X[0]:CROP_X[1],
                     CROP_Y[0]:CROP_Y[1],
                     CROP_Z[0]:CROP_Z[1]]

    # Conversione in tensore float32
    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))

    # Aggiunta della dimensione dei canali: (D, H, W) -> (1, D, H, W)
    tensor = tensor.unsqueeze(0)

    return tensor


def plot_nii_images(filepath, crop=True):
    tensor = read_nii(filepath, crop=crop)
    volume = tensor.squeeze(0).numpy()  # (D, H, W)

    d, h, w = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(volume[d // 2, :, :], cmap="viridis")
    axes[0].set_title("Sezione asse 0 (centrale)")

    axes[1].imshow(volume[:, h // 2, :], cmap="viridis")
    axes[1].set_title("Sezione asse 1 (centrale)")

    axes[2].imshow(volume[:, :, w // 2], cmap="viridis")
    axes[2].set_title("Sezione asse 2 (centrale)")

    plt.suptitle("Cropped" if crop else "Not cropped")
    plt.tight_layout()
    plt.show()


def generate_label(data_dir, output_csv, positive_prefixes=None):
    if positive_prefixes is None:
        positive_prefixes = ["ASD"]

    rows = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".nii", ".nii.gz")):
            continue

        label = 1 if any(p in fname for p in positive_prefixes) else 0
        rows.append((fname, label))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Generato {output_csv} con {len(rows)} campioni "
          f"({sum(l for _, l in rows)} positivi, "
          f"{len(rows) - sum(l for _, l in rows)} negativi).")


class NiiDataset(Dataset):

    def __init__(self, labels_csv, data_dir, crop=True, transform=None):
        self.data_dir = data_dir
        self.crop = crop
        self.transform = transform

        self.samples = []
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        filepath = os.path.join(self.data_dir, filename)

        image = read_nii(filepath, crop=self.crop)

        if self.transform:
            image = self.transform(image)

        return image, label
