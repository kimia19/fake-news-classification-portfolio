"""Reproducible, title-group-aware FA-KES train/validation/test partitions.

Keeping identical normalized titles within a single partition prevents exact
title overlap between training and held-out evaluation. Stratification is
approximate, as group constraints can affect final class proportions.
"""
import re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def normalize_title(text):
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def make_group_split(titles, labels, seed=42):
    labels = np.asarray(labels, dtype=int)
    titles = np.asarray(list(titles), dtype=object)
    if titles.ndim != 1 or len(titles) != len(labels) or len(labels) < 100:
        raise ValueError("Expected aligned, nonempty FA-KES titles and labels")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected 0=fake and 1=credible")
    groups = np.asarray([normalize_title(x) for x in titles], dtype=object)
    if any(not value for value in groups):
        raise ValueError("Empty article title is not a valid split group")

    all_indices = np.arange(len(labels))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(all_indices, labels, groups))

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(inner.split(
        train_val_idx, labels[train_val_idx], groups[train_val_idx]
    ))
    train_idx = train_val_idx[train_local]
    val_idx = train_val_idx[val_local]

    partitions = (train_idx, val_idx, test_idx)
    for idx in partitions:
        if len(idx) == 0 or set(labels[idx]) != {0, 1}:
            raise ValueError("Split is empty or missing a label")
    if len(np.concatenate(partitions)) != len(labels):
        raise AssertionError("Lost samples during group-aware splitting")

    group_sets = [set(groups[idx]) for idx in partitions]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        if group_sets[a] & group_sets[b]:
            raise AssertionError("Title group leakage detected")
    return train_idx, val_idx, test_idx, groups
