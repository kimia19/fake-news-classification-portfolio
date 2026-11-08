"""Audit an official FA-KES CSV without printing news text or committing data.

Expected source: https://zenodo.org/records/2607278 (CC BY 4.0).
Validate the original file checksum separately with download_fakes.py.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

EXPECTED_COLUMNS = {"article_title", "article_content", "labels"}
VALID_LABELS = {"0", "1"}


def audit_dataset(path: Path) -> dict:
    path = Path(path)
    data_md5 = hashlib.md5(path.read_bytes()).hexdigest()
    counts = Counter()
    missing = Counter()
    seen_titles = set()
    duplicate_titles = 0
    empty_titles = 0
    unmapped_records = 0
    unmapped_cells = 0
    sample_unmapped = []
    n = 0

    with path.open("r", encoding="latin1", newline="") as file:
        csv.field_size_limit(10**8)
        reader = csv.reader(file)
        raw_columns = next(reader)
        # The Zenodo release has four *blank*, trailing header cells:
        # unit_id,title,content,source,date,location,labels,"","","","".
        # Ignore empty spreadsheet-export columns only if their row values
        # are also empty; never deduplicate named columns silently.
        columns = [value.strip() for value in raw_columns]
        named = [name for name in columns if name]
        if len(named) != len(set(named)):
            raise ValueError(f"Duplicate named CSV columns: {named!r}")
        if not EXPECTED_COLUMNS.issubset(named):
            raise ValueError(
                "Missing expected FA-KES columns: "
                + repr(sorted(EXPECTED_COLUMNS - set(named)))
            )
        field_positions = {name: columns.index(name) for name in EXPECTED_COLUMNS}
        empty_header_positions = [i for i, name in enumerate(columns) if not name]

        for row in reader:
            if not row or not any(value.strip() for value in row):
                continue
            n += 1
            if len(row) < max(field_positions.values()) + 1:
                raise ValueError(f"Truncated CSV row {n}: only {len(row)} fields")
            extra_positions = [
                i for i, value in enumerate(row)
                if value.strip() and (i >= len(columns) or i in empty_header_positions)
            ]
            if extra_positions:
                unmapped_records += 1
                unmapped_cells += len(extra_positions)
                if len(sample_unmapped) < 10:
                    sample_unmapped.append({
                        "row": n, "csv_fields": len(row),
                        "unmapped_indices": extra_positions,
                        "unmapped_lengths": [len(row[i]) for i in extra_positions],
                    })
            label = row[field_positions["labels"]].strip()
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid binary label at row {n}: {label!r}")
            counts[label] += 1
            for col in EXPECTED_COLUMNS:
                if not row[field_positions[col]].strip():
                    missing[col] += 1
            title = row[field_positions["article_title"]].strip().casefold()
            if not title:
                empty_titles += 1
            if title in seen_titles:
                duplicate_titles += 1
            seen_titles.add(title)

    if n < 100 or set(counts) != VALID_LABELS:
        raise ValueError(f"Unexpected dataset size/class coverage: rows={n}, classes={dict(counts)}")

    return {
        "source": "https://doi.org/10.5281/zenodo.2607278",
        "md5": data_md5,
        "rows": n,
        "columns": named,
        "blank_trailing_header_columns": len(empty_header_positions),
        "label_counts": {"fake_0": counts["0"], "credible_1": counts["1"]},
        "missing": dict(sorted(missing.items())),
        "duplicate_normalized_titles": duplicate_titles,
        "empty_titles": empty_titles,
        "unmapped_records": unmapped_records,
        "unmapped_cells": unmapped_cells,
        "sample_unmapped": sample_unmapped,
        "caution": "Duplicate titles alone do not establish duplicate news articles.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/FA-KES-Dataset.csv"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = audit_dataset(args.input)
    output = json.dumps(summary, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
