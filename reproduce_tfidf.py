"""Reproduce a lightweight real-data title-only FA-KES benchmark.

Distinct from the CNN-LSTM coursework: this is a TF-IDF + Logistic Regression
sanity baseline, not a substitute for rerunning its GloVe-based neural models.

Dataset: Abu Salem et al., FA-KES (2019), https://doi.org/10.5281/zenodo.2607278
Labels: 0=fake, 1=credible. Uses original checksum and group-aware splits.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from audit_fakes import audit_dataset
from download_fakes import EXPECTED_MD5
from split_fakes import make_group_split


def build_classifier(c):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("logistic_regression", LogisticRegression(
            C=c, max_iter=1000, solver="liblinear", random_state=42
        )),
    ])


def metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classes": {
            "fake_0": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "credible_1": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def run(data_path: Path):
    audit = audit_dataset(data_path)
    if audit["md5"] != EXPECTED_MD5:
        raise ValueError("This benchmark requires the verified official Zenodo dataset")

    df = pd.read_csv(
        data_path, usecols=["article_title", "labels"],
        encoding="latin1", keep_default_na=False,
    )
    texts = df["article_title"].astype(str).to_numpy()
    labels = pd.to_numeric(df["labels"], errors="raise").to_numpy(dtype=int)
    if len(texts) != audit["rows"] or set(np.unique(labels)) != {0, 1}:
        raise AssertionError("Dataset audit and model inputs disagree")

    train_idx, val_idx, test_idx, groups = make_group_split(texts, labels, seed=42)
    candidates = {}
    for c in (0.1, 1.0, 10.0):
        model = build_classifier(c)
        model.fit(texts[train_idx], labels[train_idx])
        predicted = model.predict(texts[val_idx])
        candidates[str(c)] = float(f1_score(
            labels[val_idx], predicted, average="macro", zero_division=0
        ))

    # Lock hyperparameter choice exclusively on validation macro-F1;
    # in a tie, prefer the smaller regularization strength.
    chosen_c = min((0.1, 1.0, 10.0), key=lambda c: (-candidates[str(c)], c))
    train_val_idx = np.concatenate([train_idx, val_idx])

    selected = build_classifier(chosen_c)
    selected.fit(texts[train_val_idx], labels[train_val_idx])
    predicted_test = selected.predict(texts[test_idx])

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(texts[train_val_idx].reshape(-1, 1), labels[train_val_idx])
    dummy_predictions = dummy.predict(texts[test_idx].reshape(-1, 1))

    split_info = {}
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        split_info[name] = {
            "rows": int(len(indices)),
            "fake_0": int((labels[indices] == 0).sum()),
            "credible_1": int((labels[indices] == 1).sum()),
            "unique_normalized_titles": len(set(groups[indices])),
        }

    return {
        "experiment": "FA-KES official; TF-IDF + Logistic Regression; title-only",
        "scope": (
            "One deterministic grouped holdout sanity baseline. NOT a neural "
            "CNN-LSTM reproduction, and not an external-domain evaluation."
        ),
        "dataset": {
            "doi": audit["source"],
            "md5": audit["md5"],
            "rows": audit["rows"],
            "duplicate_normalized_titles": audit["duplicate_normalized_titles"],
            "unmapped_records_in_unused_columns": audit["unmapped_records"],
        },
        "split": {
            "method": "StratifiedGroupKFold nested 5-fold selections",
            "group": "casefolded, whitespace-normalized article title",
            "seed_outer": 42,
            "seed_inner": 43,
            "details": split_info,
            "cross_partition_title_overlap": 0,
        },
        "model_selection": {
            "criterion": "validation macro-F1 (greater is better)",
            "candidates": candidates,
            "selected_c": chosen_c,
            "final_fit": "retrained on train plus validation only",
        },
        "heldout_test": metrics(labels[test_idx], predicted_test),
        "majority_dummy_heldout_test": metrics(labels[test_idx], dummy_predictions),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/FA-KES-Dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fakes_tfidf_baseline.json"))
    args = parser.parse_args()
    result = run(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Results saved to:", args.output)
