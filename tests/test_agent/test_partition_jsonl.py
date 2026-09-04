import json
import subprocess
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
SCRIPT = Path(__file__).parents[2] / "examples" / "coding_agent_rl" / "partition_jsonl.py"


def _write_records(path: Path, count: int) -> None:
    path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(count)))


def _record_ids(path: Path) -> list[int]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


def _partition(source: Path, train: Path, eval_path: Path, *, eval_size: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--input",
            source,
            "--train",
            train,
            "--eval",
            eval_path,
            "--eval-size",
            str(eval_size),
            "--seed",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_partition_jsonl_creates_deterministic_disjoint_holdout(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_records(source, 20)

    first_train = tmp_path / "first-train.jsonl"
    first_eval = tmp_path / "first-eval.jsonl"
    second_train = tmp_path / "second-train.jsonl"
    second_eval = tmp_path / "second-eval.jsonl"
    assert _partition(source, first_train, first_eval, eval_size=5).returncode == 0
    assert _partition(source, second_train, second_eval, eval_size=5).returncode == 0

    train_ids = _record_ids(first_train)
    eval_ids = _record_ids(first_eval)
    assert len(train_ids) == 15
    assert len(eval_ids) == 5
    assert set(train_ids).isdisjoint(eval_ids)
    assert sorted(train_ids + eval_ids) == list(range(20))
    assert train_ids == sorted(train_ids)
    assert eval_ids == sorted(eval_ids)
    assert first_train.read_bytes() == second_train.read_bytes()
    assert first_eval.read_bytes() == second_eval.read_bytes()


@pytest.mark.parametrize("eval_size", [0, 4])
def test_partition_jsonl_rejects_empty_train_or_eval(tmp_path: Path, eval_size: int) -> None:
    source = tmp_path / "source.jsonl"
    _write_records(source, 4)

    result = _partition(source, tmp_path / "train.jsonl", tmp_path / "eval.jsonl", eval_size=eval_size)
    assert result.returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
