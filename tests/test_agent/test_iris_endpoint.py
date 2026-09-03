from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from slime.agent.iris_endpoint import IrisEndpoint

NUM_GPUS = 0


class FakeEndpointClient:
    def __init__(self, endpoint_id: str = "endpoint-1") -> None:
        self.endpoint_id = endpoint_id
        self.register_calls = []
        self.close_calls = 0

    def register(self, *args, **kwargs):
        self.register_calls.append((args, kwargs))
        return self.endpoint_id

    def close(self):
        self.close_calls += 1


class FakeControllerClient:
    def __init__(self) -> None:
        self.calls = []

    def mint_endpoint_token(self, request):
        self.calls.append(request)
        index = len(self.calls)
        return SimpleNamespace(
            capability_url=f"https://iris.example/proxy/token-{index}/adapter/",
            expires_at=SimpleNamespace(epoch_ms=(10_000 + index * 10_000) * 1000),
        )


class FailingEndpointClient(FakeEndpointClient):
    def register(self, *args, **kwargs):
        raise ConnectionError("controller down")


def make_endpoint(endpoint_client=None, controller_client=None):
    return IrisEndpoint(
        name="slime-adapter-test",
        address="http://10.0.0.2:18001",
        endpoint_client=endpoint_client or FakeEndpointClient(),
        controller_client=controller_client or FakeControllerClient(),
        task_attempt="task-attempt",
        link_access=2,
        ttl_seconds=3600,
        request_factory=lambda name, ttl: SimpleNamespace(
            endpoint_name=name,
            ttl=SimpleNamespace(seconds=ttl),
        ),
    )


def test_registers_link_endpoint_and_returns_capability_url():
    endpoint_client = FakeEndpointClient()
    controller_client = FakeControllerClient()
    endpoint = make_endpoint(endpoint_client, controller_client)

    with patch("slime.agent.iris_endpoint.time.time", return_value=1_000):
        assert endpoint.public_url == "https://iris.example/proxy/token-1/adapter"

    assert endpoint_client.register_calls == [
        (
            ("slime-adapter-test", "http://10.0.0.2:18001", "task-attempt", {}),
            {"access": 2},
        )
    ]
    assert controller_client.calls[0].endpoint_name == "slime-adapter-test"
    assert controller_client.calls[0].ttl.seconds == 3600


def test_refreshes_capability_url_near_expiry():
    controller_client = FakeControllerClient()
    endpoint = make_endpoint(controller_client=controller_client)

    with patch("slime.agent.iris_endpoint.time.time", return_value=1_000):
        assert "token-1" in endpoint.public_url
    with patch("slime.agent.iris_endpoint.time.time", return_value=19_500):
        assert "token-2" in endpoint.public_url

    assert len(controller_client.calls) == 2


def test_close_is_idempotent():
    endpoint_client = FakeEndpointClient()
    endpoint = make_endpoint(endpoint_client=endpoint_client)

    endpoint.close()
    endpoint.close()

    assert endpoint_client.close_calls == 1


def test_empty_endpoint_id_closes_client_and_fails():
    endpoint_client = FakeEndpointClient(endpoint_id="")

    with pytest.raises(RuntimeError, match="endpoint ID"):
        make_endpoint(endpoint_client=endpoint_client)

    assert endpoint_client.close_calls == 1


def test_registration_failure_closes_client():
    endpoint_client = FailingEndpointClient()

    with pytest.raises(ConnectionError, match="controller down"):
        make_endpoint(endpoint_client=endpoint_client)

    assert endpoint_client.close_calls == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
