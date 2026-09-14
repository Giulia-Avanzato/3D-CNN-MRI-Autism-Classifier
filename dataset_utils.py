
import random
from torch.utils.data import Subset

from mri_datautils import NiiDataset


def train_test_split_dataset(labels_csv, data_dir, split_factor=0.8,
                               crop=True, shuffle=True, seed=42):
    full_dataset = NiiDataset(labels_csv, data_dir, crop=crop)

    indices = list(range(len(full_dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    split_point = int(len(indices) * split_factor)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, test_dataset


def kfold_split_dataset(labels_csv, data_dir, k_fold=5, crop=True,
                          shuffle=True, seed=42):
    full_dataset = NiiDataset(labels_csv, data_dir, crop=crop)

    indices = list(range(len(full_dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    fold_size = len(indices) // k_fold
    folds = []

    for i in range(k_fold):
        start = i * fold_size
        # L'ultimo fold assorbe eventuali campioni rimanenti
        end = (i + 1) * fold_size if i < k_fold - 1 else len(indices)

        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        folds.append((train_dataset, test_dataset))

    return folds
