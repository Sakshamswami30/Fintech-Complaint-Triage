"""
Load the raw CFPB complaints CSV, filter rows with a narrative,
and save a stratified sample for training.

The raw file is ~8.7 GB, so we stream it in chunks to avoid
loading everything into memory at once.
"""

# ============ BLOCK 1: Imports and constants ============
import pandas as pd
import os
import sys

RAW_CSV = "data/raw/complaints.csv"
PROCESSED_DIR = "data/processed"
SAMPLE_OUT = os.path.join(PROCESSED_DIR, "complaints_sample.csv")
FILTERED_OUT = os.path.join(PROCESSED_DIR, "complaints_filtered.csv")

USECOLS = [
    "Complaint ID",
    "Product",
    "Sub-product",
    "Issue",
    "Consumer complaint narrative",
]

NARRATIVE_COL = "Consumer complaint narrative"
CHUNK_SIZE = 50_000
TARGET_SAMPLE = 200_000
SEED = 42


# ============ BLOCK 2: Ensure output folder exists ============
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============ BLOCK 3: Chunked loading + filtering ============
def load_and_filter():
    """
    Stream the raw CSV in chunks, keep only rows with a non-empty narrative,
    and write the filtered result to a CSV.
    """
    if not os.path.exists(RAW_CSV):
        sys.exit(f"Raw CSV not found at {RAW_CSV}")

    total_read = 0
    total_kept = 0
    chunk_num = 0

    if os.path.exists(FILTERED_OUT):
        os.remove(FILTERED_OUT)

    write_header = True

    reader = pd.read_csv(
        RAW_CSV,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
        dtype=str,
        on_bad_lines="skip"
    )

    for chunk in reader:
        chunk_num += 1
        total_read += len(chunk)

        chunk = chunk[chunk[NARRATIVE_COL].notna()]
        chunk = chunk[chunk[NARRATIVE_COL].str.strip() != ""]

        kept = len(chunk)
        total_kept += kept

        if kept > 0:
            chunk.to_csv(
                FILTERED_OUT,
                mode="a",
                header=write_header,
                index=False
            )
            write_header = False

        if chunk_num % 5 == 0:
            print(f"Chunks read: {chunk_num} | Rows read: {total_read:,} | Rows kept: {total_kept:,}")

    print(f"\nDone. Total rows read: {total_read:,}")
    print(f"Rows with narrative kept: {total_kept:,}")
    return FILTERED_OUT


# ============ BLOCK 4: Stratified sampling ============
def make_sample(filtered_path):
    """
    Load the filtered CSV and take a stratified sample by Product.

    Uses a simple for-loop instead of groupby().apply() because
    pandas' apply() has version-dependent behavior that can drop
    the group key from the result columns.
    """
    print(f"\nLoading filtered file for sampling: {filtered_path}")
    df = pd.read_csv(filtered_path)
    print(f"Loaded {len(df):,} rows")

    # Drop rows where Product is missing
    df = df[df["Product"].notna()]

    per_product = TARGET_SAMPLE // df["Product"].nunique()
    print(f"Target per product: ~{per_product:,}")

    # Sample each product separately, then combine.
    # Boring but correct across pandas versions.
    sampled_parts = []
    for product, group in df.groupby("Product"):
        n = min(len(group), per_product)
        sampled_parts.append(group.sample(n=n, random_state=SEED))

    sample = pd.concat(sampled_parts, ignore_index=True)

    print(f"Sample size: {len(sample):,}")
    print("\nProduct distribution in sample:")
    print(sample["Product"].value_counts())

    sample.to_csv(SAMPLE_OUT, index=False)
    print(f"\nSaved sample to {SAMPLE_OUT}")
    return sample


# ============ BLOCK 5: Main entrypoint ============
if __name__ == "__main__":
    print(f"Starting processing: {RAW_CSV}")
    print(f"Reading in chunks of {CHUNK_SIZE:,} rows\n")
    filtered_path = load_and_filter()
    make_sample(filtered_path)