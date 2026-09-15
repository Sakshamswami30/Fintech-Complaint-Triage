# sample_only.py
# Re-runs just the sampling step. Use when filtering succeeded but
# sampling failed, to avoid re-reading the 8+ GB raw file.

import sys
import os

# Make sure Python can find load_and_sample in this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_and_sample import make_sample


if __name__ == "__main__":
    make_sample("data/processed/complaints_filtered.csv")