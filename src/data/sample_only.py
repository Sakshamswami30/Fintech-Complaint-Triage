"""
Re-run sampling only. Assumes complaints_filtered.csv already exists.

Use this when the filtering step succeeded but the sampling step failed,
so you don't have to re-read the 8.7 GB raw CSV.
"""
import sys
import os

# Make sure Python can find load_and_sample in this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_and_sample import make_sample


if __name__ == "__main__":
    make_sample("data/processed/complaints_filtered.csv")