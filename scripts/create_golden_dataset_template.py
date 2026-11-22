"""Create the human annotation worksheet for the Phase 4 Golden Dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.golden_dataset import write_annotation_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/golden_dataset/annotations.json"))
    parser.add_argument("--questions", type=int, default=25, choices=range(20, 31))
    args = parser.parse_args()
    write_annotation_template(args.output, args.questions)
    print(f"Created {args.questions} empty annotation slots at {args.output}")


if __name__ == "__main__":
    main()
