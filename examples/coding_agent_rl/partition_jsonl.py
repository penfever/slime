"""Create deterministic, disjoint training and evaluation JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path


def _nonempty_lines(path: Path):
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                yield line


def partition_jsonl(
    input_path: Path,
    train_path: Path,
    eval_path: Path,
    *,
    eval_size: int,
    seed: int,
    train_size: int | None,
    train_seed: int | None,
) -> None:
    """Partition JSONL records by a stable content hash.

    Args:
        input_path: Source JSONL file.
        train_path: Destination for records not selected for evaluation.
        eval_path: Destination for evaluation records.
        eval_size: Exact number of evaluation records.
        seed: Salt for deterministic selection.
        train_size: Optional exact number of training records to retain.
        train_seed: Salt for deterministic training-subset selection.
    """
    resolved_paths = {input_path.resolve(), train_path.resolve(), eval_path.resolve()}
    if len(resolved_paths) != 3:
        raise ValueError("input, train, and eval paths must be distinct")

    seed_bytes = str(seed).encode()
    ranked_records = [
        (hashlib.sha256(seed_bytes + b"\0" + line).digest(), index)
        for index, line in enumerate(_nonempty_lines(input_path))
    ]
    if not 0 < eval_size < len(ranked_records):
        raise ValueError(f"eval_size must be between 1 and {len(ranked_records) - 1}")

    eval_indices = {index for _, index in sorted(ranked_records)[:eval_size]}
    train_indices = {index for _, index in ranked_records if index not in eval_indices}
    if train_size is not None:
        if train_seed is None:
            raise ValueError("train_seed is required when train_size is set")
        if not 0 < train_size <= len(train_indices):
            raise ValueError(f"train_size must be between 1 and {len(train_indices)}")
        train_seed_bytes = str(train_seed).encode()
        ranked_train_records = [
            (hashlib.sha256(train_seed_bytes + b"\0" + line).digest(), index)
            for index, line in enumerate(_nonempty_lines(input_path))
            if index in train_indices
        ]
        train_indices = {index for _, index in sorted(ranked_train_records)[:train_size]}
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    train_temp = tempfile.NamedTemporaryFile(dir=train_path.parent, delete=False)
    eval_temp = tempfile.NamedTemporaryFile(dir=eval_path.parent, delete=False)
    try:
        with train_temp, eval_temp:
            for index, line in enumerate(_nonempty_lines(input_path)):
                if index in eval_indices:
                    destination = eval_temp
                elif index in train_indices:
                    destination = train_temp
                else:
                    continue
                destination.write(line if line.endswith(b"\n") else line + b"\n")
        os.replace(train_temp.name, train_path)
        os.replace(eval_temp.name, eval_path)
    except BaseException:
        Path(train_temp.name).unlink(missing_ok=True)
        Path(eval_temp.name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--eval-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--train-seed", type=int)
    args = parser.parse_args()
    partition_jsonl(
        args.input,
        args.train,
        args.eval,
        eval_size=args.eval_size,
        seed=args.seed,
        train_size=args.train_size,
        train_seed=args.train_seed,
    )


if __name__ == "__main__":
    main()
