import ast
import os
import sys

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "runtime.txt",
    ".streamlit/config.toml",
    "Data/data/flipkart_com-ecommerce_sample.csv",
    "embeddings.npy",
    "id_list.npy",
]


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def main():
    missing = [path for path in REQUIRED_FILES if not os.path.exists(path)]
    if missing:
        fail("missing required file(s): " + ", ".join(missing))

    catalog = pd.read_csv("Data/data/flipkart_com-ecommerce_sample.csv")
    embeddings = np.load("embeddings.npy")
    ids = np.load("id_list.npy", allow_pickle=True)

    if embeddings.ndim != 2:
        fail(f"embeddings must be a 2D matrix, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(ids):
        fail(f"embedding rows ({embeddings.shape[0]}) do not match ids ({len(ids)})")
    if len(catalog) != len(ids):
        fail(f"catalog rows ({len(catalog)}) do not match ids ({len(ids)})")

    required_columns = {"uniq_id", "product_name", "description", "image"}
    missing_columns = required_columns.difference(catalog.columns)
    if missing_columns:
        fail("catalog missing column(s): " + ", ".join(sorted(missing_columns)))

    sample_images = catalog["image"].dropna().head(25)
    for value in sample_images:
        try:
            ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            fail(f"image field is not parseable: {exc}")

    print("OK: deployment artifacts are present and aligned")
    print(f"Catalog rows: {len(catalog):,}")
    print(f"Embedding shape: {embeddings.shape[0]:,} x {embeddings.shape[1]}")
    print(f"Unique ids: {len(ids):,}")


if __name__ == "__main__":
    main()
