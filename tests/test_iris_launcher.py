import json
from pathlib import Path
import sys

import fsspec
import pytest

from infra.iris.file_transfer import S3Input, run as run_file_transfer
from infra.iris.launcher import IrisLaunchSpec, LaunchConfigError, run


NUM_GPUS = 0


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def submit(self, spec, workspace, *, wait):
        self.calls.append((spec, workspace, wait))
        return "/test/slime-job"


def base_spec(**overrides):
    values = {
        "job_name": "slime-test",
        "task_image": "registry.example/slime@sha256:abc",
        "command": ("python", "train.py"),
        "cluster": "cw-rno2a",
    }
    values.update(overrides)
    return IrisLaunchSpec(**values)


def test_spec_requires_rendezvous_for_multiple_iris_tasks():
    with pytest.raises(LaunchConfigError, match="rendezvous-dir"):
        base_spec(nodes=2).validate()


def test_multinode_dry_run_uses_rank_aware_ray_runtime(tmp_path, capsys):
    exit_code = run(
        [
            "--cluster",
            "cw-rno2a",
            "--task-image",
            "image",
            "--nodes",
            "5",
            "--rendezvous-dir",
            "s3://experiments/slime-test",
            "--workspace",
            str(tmp_path),
            "--dry-run",
            "--",
            "python",
            "train.py",
        ]
    )

    rendered = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert rendered["nodes"] == 5
    assert rendered["task_command"][-2:] == ["python", "train.py"]
    assert rendered["task_command"][:3] == ["python", "-m", "infra.iris.ray_runtime"]


def test_multinode_s3_inputs_materialize_before_ray(tmp_path, capsys):
    exit_code = run(
        [
            "--cluster",
            "cw-rno2a",
            "--task-image",
            "image",
            "--nodes",
            "5",
            "--rendezvous-dir",
            "s3://experiments/slime-test",
            "--s3-input",
            "s3://assets/model=/app/model",
            "--workspace",
            str(tmp_path),
            "--dry-run",
            "--",
            "python",
            "train.py",
        ]
    )

    rendered = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert rendered["s3_inputs"] == ["s3://assets/model=/app/model"]
    assert rendered["task_command"] == [
        "python",
        "-m",
        "infra.iris.file_transfer",
        "--s3-input",
        "s3://assets/model=/app/model",
        "--",
        "python",
        "-m",
        "infra.iris.ray_runtime",
        "--rendezvous-dir",
        "s3://experiments/slime-test",
        "--",
        "python",
        "train.py",
    ]


def test_s3_input_preserves_equals_in_object_key():
    item = S3Input.parse("s3://bucket/tmp/ttl=14d/assets=/app/assets")

    assert item.source == "s3://bucket/tmp/ttl=14d/assets"
    assert item.destination == Path("/app/assets")


def test_s3_input_tree_replaces_destination_before_command(tmp_path, monkeypatch):
    filesystem = fsspec.filesystem("memory")
    filesystem.pipe("bucket/assets/model/config.json", b'{"model": "qwen"}')
    filesystem.pipe("bucket/assets/data/train.jsonl", b"one\ntwo\n")
    storage_options = {}

    def resolve_filesystem(_source, **options):
        storage_options.update(options)
        return filesystem, "bucket/assets"

    monkeypatch.setattr(fsspec.core, "url_to_fs", resolve_filesystem)
    destination = tmp_path / "assets"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale")

    exit_code = run_file_transfer(
        [
            "--s3-input",
            f"s3://bucket/assets={destination}",
            "--",
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; root = Path(sys.argv[1]); "
                "assert (root / 'model/config.json').read_text() == '{\"model\": \"qwen\"}'; "
                "assert (root / 'data/train.jsonl').read_text() == 'one\\ntwo\\n'; "
                "assert not (root / 'stale.txt').exists()"
            ),
            str(destination),
        ]
    )

    assert exit_code == 0
    assert storage_options["config_kwargs"]["s3"] == {"addressing_style": "virtual"}


@pytest.mark.parametrize(
    "assignment",
    [
        "s3://bucket=/app/assets",
        "s3://bucket/assets=relative/path",
        "s3://bucket/assets=/",
    ],
)
def test_s3_input_rejects_unsafe_transfer_boundaries(assignment):
    with pytest.raises(ValueError):
        S3Input.parse(assignment)


def test_spec_rejects_overlapping_s3_input_destinations():
    with pytest.raises(LaunchConfigError, match="must not overlap"):
        base_spec(
            s3_inputs=(
                S3Input.parse("s3://assets/model=/app/assets"),
                S3Input.parse("s3://assets/config=/app/assets/config"),
            )
        ).validate()


def test_spec_requires_exactly_one_controller_selection():
    with pytest.raises(LaunchConfigError, match="select exactly one"):
        base_spec(cluster=None).validate()

    with pytest.raises(LaunchConfigError, match="select exactly one"):
        base_spec(controller_url="https://iris.example").validate()


def test_spec_rejects_invalid_direct_api_values():
    with pytest.raises(LaunchConfigError, match="priority"):
        base_spec(priority="urgent").validate()

    with pytest.raises(LaunchConfigError, match="environment variable"):
        base_spec(env={"NOT-AN-ENV": "value"}).validate()


def test_secret_resolution_is_explicit_and_reports_missing_names():
    spec = base_spec(env={"PUBLIC": "visible"}, secret_env_names=("TOKEN",))
    assert spec.resolved_env({"TOKEN": "secret"}) == {"PUBLIC": "visible", "TOKEN": "secret"}

    with pytest.raises(LaunchConfigError, match="TOKEN"):
        spec.resolved_env({})


def test_redacted_request_contains_no_environment_values():
    spec = base_spec(env={"PUBLIC": "sensitive-ish"}, secret_env_names=("TOKEN",))
    rendered = spec.redacted_dict()

    assert rendered["env"] == ["PUBLIC"]
    assert rendered["secret_env_names"] == ["TOKEN"]
    assert "sensitive-ish" not in str(rendered)


def test_run_submits_validated_command_without_importing_iris(tmp_path):
    backend = RecordingBackend()
    exit_code = run(
        [
            "--cluster",
            "cw-rno2a",
            "--task-image",
            "registry.example/slime@sha256:abc",
            "--job-name",
            "slime-test",
            "--workspace",
            str(tmp_path),
            "--env",
            "RUN_NAME=test",
            "--no-wait",
            "--",
            "python",
            "train.py",
            "--actor-num-nodes",
            "1",
        ],
        backend=backend,
    )

    assert exit_code == 0
    assert len(backend.calls) == 1
    spec, workspace, wait = backend.calls[0]
    assert spec.command == ("python", "train.py", "--actor-num-nodes", "1")
    assert spec.env == {"RUN_NAME": "test"}
    assert workspace == Path(tmp_path)
    assert wait is False


def test_run_rejects_duplicate_public_environment_names(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        run(
            [
                "--cluster",
                "cw-rno2a",
                "--task-image",
                "image",
                "--workspace",
                str(tmp_path),
                "--env",
                "NAME=first",
                "--env",
                "NAME=second",
                "--",
                "true",
            ]
        )

    assert error.value.code == 2
    assert "provided more than once" in capsys.readouterr().err


def test_dry_run_checks_secret_presence_but_redacts_value(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("API_TOKEN", "do-not-print")
    exit_code = run(
        [
            "--controller-url",
            "https://iris.example",
            "--task-image",
            "image",
            "--workspace",
            str(tmp_path),
            "--secret-env",
            "API_TOKEN",
            "--dry-run",
            "--",
            "true",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "API_TOKEN" in output
    assert "do-not-print" not in output


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
