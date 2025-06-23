# data_utils.py

import os
import zipfile
import shutil
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

def unzip_images(zip_path: str, extract_to: str="data/images"):
    """Unzip images ZIP into a folder and return that folder path."""
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    return extract_to

def read_csv_auto(path: str) -> pd.DataFrame:
    """Read a CSV with automatic delimiter sniffing (handles , ; or \\t)."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(2048)
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t'])
        f.seek(0)
        return pd.read_csv(f, sep=dialect.delimiter, engine='python')

def load_csvs(csv_paths: list[str]) -> pd.DataFrame:
    """
    Load multiple CSVs (auto-detect their delimiters), concatenate them,
    then rename the first matching filename column to 'filename'.
    """
    dfs = [ read_csv_auto(p) for p in csv_paths ]
    df = pd.concat(dfs, ignore_index=True)

    candidates = [
        c for c in df.columns
        if c.lower() in ('filename','file','nom panneau','image','image name')
    ]
    if not candidates:
        raise ValueError(
            "❌ No column matched one of "
            "['filename','file','nom panneau','image','image name']. "
            "Please rename your CSV header accordingly."
        )

    return df.rename(columns={candidates[0]: 'filename'})

def split_data(df: pd.DataFrame,
               test_size: float = 0.2,
               stratify_col: str | None = 'category',
               random_state: int = 42) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Stratified train/test split (if `category` exists), else random split."""
    if stratify_col and stratify_col in df.columns:
        return train_test_split(
            df,
            test_size=test_size,
            stratify=df[stratify_col],
            random_state=random_state
        )
    else:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
# In data_utils.py, replace your existing export_to_folders with this:

def export_to_folders(df: pd.DataFrame,
                      images_dir: str,
                      output_dir: str = 'output',
                      split_name: str = 'train'):
    """
    Copy each row’s image into output/{split_name}/{category}/...
    Builds an index once, then copies matched files.
    Falls back by copying *all* images into 'unknown' if no class folders created.
    """
    import shutil

    split_dir = os.path.join(output_dir, split_name)
    # clean & recreate split_dir root
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)

    # 1) index all images by base filename
    fname_to_path = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                base, _ = os.path.splitext(f)
                fname_to_path.setdefault(base, os.path.join(root, f))

    # 2) copy according to df
    for _, row in df.iterrows():
        raw = row.get('filename', None)
        if pd.isna(raw):
            continue
        base = str(raw)
        # strip “.0” from things like “123.0”
        if base.endswith('.0') and base[:-2].isdigit():
            base = base[:-2]

        src = fname_to_path.get(base)
        if not src:
            continue

        label = str(row.get('category', 'unknown'))
        dest = os.path.join(split_dir, label)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(src, dest)

    # 3) fallback: if no class subfolders were made, copy everything into 'unknown'
    subs = [d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))]
    if not subs:
        unk_dir = os.path.join(split_dir, 'unknown')
        os.makedirs(unk_dir, exist_ok=True)
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(os.path.join(root, f), unk_dir)
