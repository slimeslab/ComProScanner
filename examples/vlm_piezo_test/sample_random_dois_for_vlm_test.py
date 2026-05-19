"""
sample_random_dois_for_vlm_test.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 15-04-2026
"""

from __future__ import annotations

import random
from pathlib import Path


SEED = 42
SAMPLE_SIZE = 50


def main() -> None:
    here = Path(__file__).resolve().parent
    in_path = here / "possible_dois_for_vlm_test.txt"
    out_path = here / "random_dois_for_vlm_test.txt"

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as handle:
        dois = [line.strip() for line in handle if line.strip()]

    random.seed(SEED)
    sample_size = min(SAMPLE_SIZE, len(dois))
    sampled = random.sample(dois, sample_size) if sample_size else []

    with out_path.open("w", encoding="utf-8") as handle:
        for doi in sampled:
            handle.write(f"{doi}\n")
            print(doi)

    print(f"\nInput DOIs: {len(dois)}")
    print(f"Sampled DOIs: {len(sampled)}")
    print(f"Saved sample to: {out_path}")


if __name__ == "__main__":
    main()
