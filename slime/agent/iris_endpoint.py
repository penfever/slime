"""Publish an in-task HTTP service through an Iris capability endpoint."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable
from typing import Any


DEFAULT_TOKEN_TTL_SECONDS = 24 * 60 * 60
TOKEN_REFRESH_MARGIN_SECONDS = 15 * 60


def _mint_request(endpoint_name: str, ttl_seconds: int) -> Any:
    from iris.rpc import controller_pb2
    from iris.time_proto import duration_to_proto
    from rigging.timing import Duration

    return controller_pb2.Controller.MintEndpointTokenRequest(
        endpoint_name=endpoint_name,
        ttl=duration_to_proto(Duration.from_seconds(ttl_seconds)),
    )


class IrisEndpoint:
    """A leased Iris endpoint with a refreshable public capability URL."""

    def __init__(
        self,
        *,
        name: str,
        address: str,
        endpoint_client: Any,
        controller_client: Any,
        task_attempt: Any,
        link_access: int,
        ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
        request_factory: Callable[[str, int], Any] = _mint_request,
    ) -> None:
        self._name = name
        self._endpoint_client = endpoint_client
        self._controller_client = controller_client
        self._ttl_seconds = ttl_seconds
        self._request_factory = request_factory
        self._lock = threading.Lock()
        self._closed = False
        self._capability_url: str | None = None
        self._expires_at = 0.0

        try:
            endpoint_id = endpoint_client.register(name, address, task_attempt, {}, access=link_access)
        except BaseException:
            endpoint_client.close()
            raise
        if not endpoint_id:
            endpoint_client.close()
            raise RuntimeError(f"Iris did not return an endpoint ID for {name!r}")

    @property
    def public_url(self) -> str:
        """Return a capability URL, refreshing it before the token expires."""
        with self._lock:
            now = time.time()
            if self._capability_url is not None and self._expires_at - now > TOKEN_REFRESH_MARGIN_SECONDS:
                return self._capability_url

            response = self._controller_client.mint_endpoint_token(
                self._request_factory(self._name, self._ttl_seconds)
            )
            capability_url = response.capability_url.rstrip("/")
            if not capability_url:
                raise RuntimeError(f"Iris did not return a capability URL for {self._name!r}")
            self._capability_url = capability_url
            self._expires_at = response.expires_at.epoch_ms / 1000.0
            return capability_url

    def close(self) -> None:
        """Stop renewing the endpoint lease and unregister it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._endpoint_client.close()


def publish_iris_endpoint(port: int) -> IrisEndpoint:
    """Register a local port from the current Iris task as a LINK endpoint."""
    try:
        from iris.cluster.client.endpoint_client import EndpointClient
        from iris.cluster.client.job_info import get_job_info
        from iris.cluster.types import EndpointAccess
        from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
        from iris.rpc.controller_connect import ControllerServiceClientSync, EndpointServiceClientSync
    except ImportError as exc:
        raise RuntimeError(
            "automatic adapter publication requires marin-iris; install it or set "
            "ADAPTER_PUBLIC_URL/ADAPTER_PUBLIC_HOST explicitly"
        ) from exc

    info = get_job_info()
    if info is None or not info.controller_address:
        raise RuntimeError(
            "automatic adapter publication requires an Iris task; set "
            "ADAPTER_PUBLIC_URL/ADAPTER_PUBLIC_HOST when running elsewhere"
        )

    task_identity = os.environ.get("IRIS_TASK_ID", str(info.task_attempt))
    suffix = hashlib.sha256(f"{task_identity}:{os.getpid()}:{port}".encode()).hexdigest()[:16]
    name = f"slime-adapter-{suffix}"
    address = f"http://{info.advertise_host}:{port}"
    endpoint_stub = EndpointServiceClientSync(
        info.controller_address,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )
    controller_stub = ControllerServiceClientSync(
        info.controller_address,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )
    return IrisEndpoint(
        name=name,
        address=address,
        endpoint_client=EndpointClient(endpoint_stub),
        controller_client=controller_stub,
        task_attempt=info.task_attempt,
        link_access=EndpointAccess.ENDPOINT_ACCESS_LINK,
    )
