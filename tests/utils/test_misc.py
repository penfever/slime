import sys
from types import SimpleNamespace

import pytest

from slime.utils.misc import get_current_node_ip

NUM_GPUS = 0


def _fake_ray(*, initialized, node_id="current", nodes=(), fallback="127.0.0.1"):
    return SimpleNamespace(
        is_initialized=lambda: initialized,
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: node_id),
        nodes=lambda: list(nodes),
        _private=SimpleNamespace(
            services=SimpleNamespace(get_node_ip_address=lambda: fallback),
        ),
    )


def test_get_current_node_ip_prefers_runtime_node_address(monkeypatch):
    ray = _fake_ray(
        initialized=True,
        nodes=(
            {"NodeID": "other", "NodeManagerAddress": "10.0.0.2"},
            {"NodeID": "current", "NodeManagerAddress": "10.0.0.3"},
        ),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)

    assert get_current_node_ip() == "10.0.0.3"


def test_get_current_node_ip_falls_back_to_host_probe(monkeypatch):
    ray = _fake_ray(initialized=True, nodes=({"NodeID": "other"},), fallback="[2001:db8::1]")
    monkeypatch.setitem(sys.modules, "ray", ray)

    assert get_current_node_ip() == "2001:db8::1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
