"""Convert TaskTrove parquet archives containing Harbor tasks to Slime JSONL."""

from __future__ import annotations

import argparse
import io
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from slime.agent.harbor_converter import ConversionOverrides, HarborConversionError, convert_task, write_rows


def _safe_members(archive: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    """Yield non-oracle regular entries whose paths cannot escape extraction."""
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise HarborConversionError(f"Unsafe TaskTrove archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise HarborConversionError(f"Unsupported TaskTrove archive member: {member.name!r}")
        if not path.parts or path.parts[0] != "solution":
            yield member


def _archive_name(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith(".tar.gz"):
        raise HarborConversionError(f"TaskTrove path must end in .tar.gz, got {value!r}")
    name = value.removesuffix(".tar.gz")
    if not name or PurePosixPath(value).name != value:
        raise HarborConversionError(f"TaskTrove path must be a plain archive name, got {value!r}")
    return name


def _extract_task(archive: tarfile.TarFile, task_dir: Path) -> None:
    """Extract validated files without reading the archive's solution subtree."""
    for member in _safe_members(archive):
        destination = task_dir / member.name
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        source = archive.extractfile(member)
        if source is None:
            raise HarborConversionError(f"Could not read TaskTrove archive member: {member.name!r}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read())
        destination.chmod(member.mode & 0o777)


def _convert_source(source: dict[str, Any], overrides: ConversionOverrides, seen: set[str]) -> dict[str, Any]:
    name = _archive_name(source["path"])
    if name in seen:
        raise HarborConversionError(f"TaskTrove archive name is not unique: {name!r}")
    seen.add(name)
    payload = source["task_binary"]
    if not isinstance(payload, bytes):
        raise HarborConversionError(f"TaskTrove task_binary for {name!r} is not bytes")
    with tempfile.TemporaryDirectory() as temporary_dir:
        task_dir = Path(temporary_dir) / name
        task_dir.mkdir()
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                _extract_task(archive, task_dir)
        except tarfile.TarError as error:
            raise HarborConversionError(f"Could not extract TaskTrove archive {name!r}: {error}") from error
        return convert_task(task_dir, overrides)


def convert_tasktrove_parquet(
    parquet_path: Path,
    output_path: Path,
    overrides: ConversionOverrides,
) -> int:
    """Convert every path/task_binary row atomically and return the row count."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise HarborConversionError("TaskTrove conversion requires pyarrow") from error

    parquet = pq.ParquetFile(parquet_path)
    required = {"path", "task_binary"}
    missing = required.difference(parquet.schema.names)
    if missing:
        raise HarborConversionError(f"TaskTrove parquet is missing columns: {', '.join(sorted(missing))}")

    seen: set[str] = set()

    def rows() -> Iterator[dict[str, Any]]:
        for batch in parquet.iter_batches(columns=["path", "task_binary"]):
            for source in batch.to_pylist():
                yield _convert_source(source, overrides, seen)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return write_rows(rows(), output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    sandbox = parser.add_mutually_exclusive_group(required=True)
    sandbox.add_argument("--image")
    sandbox.add_argument("--snapshot")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-file", action="append", default=[], metavar="PATH")
    args = parser.parse_args(argv)
    try:
        count = convert_tasktrove_parquet(
            args.parquet,
            args.output,
            ConversionOverrides(
                image=args.image,
                snapshot=args.snapshot,
                workdir=args.workdir,
                output_files=tuple(args.output_file),
            ),
        )
    except HarborConversionError as error:
        parser.error(str(error))
    print(f"Wrote {count} TaskTrove tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
