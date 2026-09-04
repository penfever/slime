"""Materialize S3 inputs and mirror task outputs while running an Iris command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


S3_CONNECT_TIMEOUT = 10
S3_READ_TIMEOUT = 60
S3_MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class S3Input:
    """An S3 object or prefix and its absolute node-local destination."""

    source: str
    destination: Path

    @classmethod
    def parse(cls, value: str) -> S3Input:
        source, separator, destination = value.rpartition("=")
        bucket, path_separator, source_path = source.removeprefix("s3://").partition("/")
        if not separator or not source.startswith("s3://") or not bucket or not path_separator or not source_path:
            raise ValueError("expected s3://BUCKET/PATH=/ABSOLUTE/LOCAL/PATH")
        local_path = Path(destination)
        if not local_path.is_absolute():
            raise ValueError("S3 input destination must be an absolute path")
        if local_path == Path(local_path.anchor):
            raise ValueError("S3 input destination must not be the filesystem root")
        return cls(source=source.rstrip("/"), destination=local_path)

    def as_assignment(self) -> str:
        return f"{self.source}={self.destination}"


@dataclass(frozen=True)
class S3Output:
    """An absolute node-local source mirrored into an S3 object prefix."""

    source: Path
    destination: str

    @classmethod
    def parse(cls, value: str) -> S3Output:
        source, separator, destination = value.partition("=")
        bucket, path_separator, destination_path = destination.removeprefix("s3://").partition("/")
        local_path = Path(source)
        if (
            not separator
            or not destination.startswith("s3://")
            or not bucket
            or not path_separator
            or not destination_path
        ):
            raise ValueError("expected /ABSOLUTE/LOCAL/PATH=s3://BUCKET/PATH")
        if not local_path.is_absolute():
            raise ValueError("S3 output source must be an absolute path")
        if local_path == Path(local_path.anchor):
            raise ValueError("S3 output source must not be the filesystem root")
        return cls(source=local_path, destination=destination.rstrip("/"))

    def as_assignment(self) -> str:
        return f"{self.source}={self.destination}"


def _filesystem_and_path(uri: str):
    try:
        import fsspec  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("S3 transfers require fsspec and s3fs in the task image") from error

    try:
        return fsspec.core.url_to_fs(
            uri,
            config_kwargs={
                "connect_timeout": S3_CONNECT_TIMEOUT,
                "read_timeout": S3_READ_TIMEOUT,
                "retries": {"max_attempts": S3_MAX_ATTEMPTS, "mode": "standard"},
                "s3": {"addressing_style": "virtual"},
            },
        )
    except ImportError as error:
        raise RuntimeError("S3 transfers require fsspec and s3fs in the task image") from error


def _replace(staging: Path, destination: Path) -> None:
    backup: Path | None = None
    try:
        if destination.exists() or destination.is_symlink():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.old-", dir=destination.parent))
            backup.rmdir()
            os.replace(destination, backup)
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if staging.exists():
            if staging.is_dir():
                shutil.rmtree(staging)
            else:
                staging.unlink()
        if backup is not None and backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()


def materialize_s3_input(item: S3Input) -> None:
    """Atomically replace a node-local path with one S3 object or prefix."""
    filesystem, source_path = _filesystem_and_path(item.source)
    item.destination.parent.mkdir(parents=True, exist_ok=True)

    if filesystem.isfile(source_path):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{item.destination.name}.staging-", dir=item.destination.parent
        )
        os.close(descriptor)
        staging = Path(temporary_name)
        try:
            filesystem.get_file(source_path, str(staging))
            expected_size = int(filesystem.info(source_path)["size"])
            if staging.stat().st_size != expected_size:
                raise ValueError(f"S3 input size mismatch for {item.source}")
            _replace(staging, item.destination)
        finally:
            staging.unlink(missing_ok=True)
        return

    source_root = PurePosixPath("/", source_path.lstrip("/"))
    source_files = sorted(path for path in filesystem.find(source_path) if not filesystem.isdir(path))
    if not source_files:
        raise ValueError(f"S3 input contains no files: {item.source}")

    staging = Path(tempfile.mkdtemp(prefix=f".{item.destination.name}.staging-", dir=item.destination.parent))
    try:
        for source_file in source_files:
            normalized_source = PurePosixPath("/", source_file.lstrip("/"))
            relative = normalized_source.relative_to(source_root)
            local_file = staging.joinpath(*relative.parts)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            filesystem.get_file(source_file, str(local_file))
            expected_size = int(filesystem.info(source_file)["size"])
            if local_file.stat().st_size != expected_size:
                raise ValueError(f"S3 input size mismatch for {item.source}/{relative}")
        _replace(staging, item.destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def mirror_s3_output(item: S3Output, uploaded: dict[Path, tuple[int, int]] | None = None) -> int:
    """Upload new or changed files without deleting objects written by other tasks."""
    if not item.source.exists():
        return 0

    filesystem, destination_path = _filesystem_and_path(item.destination)
    local_files = (
        [item.source] if item.source.is_file() else sorted(path for path in item.source.rglob("*") if path.is_file())
    )
    uploaded = uploaded if uploaded is not None else {}
    uploaded_count = 0
    for local_file in local_files:
        stat = local_file.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if uploaded.get(local_file) == fingerprint:
            continue
        relative = Path(local_file.name) if item.source.is_file() else local_file.relative_to(item.source)
        remote_file = f"{destination_path.rstrip('/')}/{relative.as_posix()}"
        filesystem.put_file(str(local_file), remote_file)
        remote_size = int(filesystem.info(remote_file)["size"])
        if remote_size != stat.st_size:
            raise ValueError(f"S3 output size mismatch for {item.destination}/{relative.as_posix()}")
        uploaded[local_file] = fingerprint
        uploaded_count += 1
    return uploaded_count


def parse_s3_input_argument(value: str) -> S3Input:
    """Parse an S3 input for either the launcher or task-runtime CLI."""
    try:
        return S3Input.parse(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_s3_output_argument(value: str) -> S3Output:
    """Parse an S3 output for either the launcher or task-runtime CLI."""
    try:
        return S3Output.parse(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-input", action="append", type=parse_s3_input_argument, default=[])
    parser.add_argument("--s3-output", action="append", type=parse_s3_output_argument, default=[])
    parser.add_argument("--s3-sync-interval-seconds", type=float, default=300.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a task command is required after '--'")
    if args.s3_sync_interval_seconds <= 0:
        raise ValueError("--s3-sync-interval-seconds must be greater than zero")
    for item in args.s3_input:
        materialize_s3_input(item)
    if not args.s3_output:
        return subprocess.run(command, check=False).returncode

    uploaded = {item: {} for item in args.s3_output}
    process = subprocess.Popen(command)
    while True:
        try:
            return_code = process.wait(timeout=args.s3_sync_interval_seconds)
            break
        except subprocess.TimeoutExpired:
            for item in args.s3_output:
                try:
                    mirror_s3_output(item, uploaded[item])
                except (OSError, RuntimeError, ValueError) as error:
                    print(f"S3 output sync failed; will retry: {error}", file=sys.stderr, flush=True)

    output_failed = False
    for item in args.s3_output:
        try:
            mirror_s3_output(item, uploaded[item])
        except (OSError, RuntimeError, ValueError) as error:
            print(f"final S3 output sync failed: {error}", file=sys.stderr, flush=True)
            output_failed = True
    return return_code if return_code != 0 else int(output_failed)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
