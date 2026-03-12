import argparse
from datetime import datetime, UTC

import numpy as np
from tqdm import tqdm

from src.config import (
    DEFAULT_COUNT_PER_SIZE,
    DEFAULT_SIZES,
    GENERATION_RETRIES_PER_PUZZLE,
    OUTPUT_DIR,
    RNG_SEED,
)
from src.exporter import export_manifest, export_puzzle
from src.generator import generate_valid_grid
from src.utils import ensure_dir


def _bounded_size(value: str) -> int:
    try:
        size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid size: {value}") from exc

    if size < 5 or size > 100:
        raise argparse.ArgumentTypeError(
            f"Size {size} is out of range. Valid range is 5..100."
        )

    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nonogram dataset generator")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=_bounded_size,
        default=DEFAULT_SIZES,
        help="List of puzzle sizes (5..100), e.g. --sizes 5 10 15",
    )
    parser.add_argument(
        "--count-per-size",
        type=int,
        default=DEFAULT_COUNT_PER_SIZE,
        help="How many puzzles to generate for each size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help="Random seed for reproducible generation",
    )
    args = parser.parse_args()
    invalid = [size for size in args.sizes if size < 5 or size > 100]
    if invalid:
        parser.error(f"--sizes values must be in 5..100. Invalid: {invalid}")
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)

    rng = np.random.default_rng(args.seed)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "count_per_size": args.count_per_size,
        "sizes": {},
        "total_puzzles": 0,
    }

    for size in args.sizes:
        print(f"\nGenerating {args.count_per_size} puzzles for size {size}x{size}...")
        generated = 0

        for index in tqdm(range(1, args.count_per_size + 1), desc=f"{size}x{size}"):
            grid = None
            for attempt in range(1, GENERATION_RETRIES_PER_PUZZLE + 1):
                try:
                    grid = generate_valid_grid(size=size, rng=rng)
                    break
                except RuntimeError as exc:
                    if attempt < GENERATION_RETRIES_PER_PUZZLE:
                        tqdm.write(
                            f"[warn] {size}x{size} puzzle {index:04d} failed "
                            f"(attempt {attempt}/{GENERATION_RETRIES_PER_PUZZLE}); retrying."
                        )
                    else:
                        tqdm.write(
                            f"[warn] {size}x{size} puzzle {index:04d} skipped after "
                            f"{GENERATION_RETRIES_PER_PUZZLE} retries: {exc}"
                        )

            if grid is None:
                continue

            export_puzzle(
                grid=grid,
                size=size,
                index=index,
                output_root=OUTPUT_DIR,
            )
            generated += 1

        manifest["sizes"][f"{size}x{size}"] = generated
        manifest["total_puzzles"] += generated

    export_manifest(OUTPUT_DIR, manifest)

    print("\nDataset generation completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
