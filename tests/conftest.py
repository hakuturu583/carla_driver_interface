"""The transports a policy is reached over, so every wire test runs on each.

``tcp`` and ``unix`` are real gRPC servers (:func:`~carla_driver_interface.server.serving`,
on a TCP port and on a Unix domain socket); ``in_process`` is
:class:`~carla_driver_interface.inprocess.InProcessChannel`, which must behave as
they do -- same messages, same bytes, same errors.
"""

from __future__ import annotations

import shutil
import tempfile
from concurrent import futures
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import grpc
import pytest

from carla_driver_interface.driver import BaseDriver
from carla_driver_interface.inprocess import InProcessChannel, in_process_channel
from carla_driver_interface.protocol import channel_options
from carla_driver_interface.server import serving

TRANSPORTS = ("tcp", "unix", "in_process")


class Transport:
    """How a test reaches a policy: :meth:`serve` yields a channel to it."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Transport({self.name!r})"

    @contextmanager
    def _socket_dir(self) -> Iterator[str]:
        # Short: a Unix socket path must fit in 108 bytes, which pytest's
        # tmp_path for a parametrized test may not.
        directory = tempfile.mkdtemp(prefix="cdi-")
        try:
            yield directory
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    @contextmanager
    def serve(self, policy: BaseDriver, mode: str = "carla") -> Iterator[grpc.Channel]:
        """Serve ``policy`` as a runtime would reach it; yields the client's channel."""
        if self.name == "in_process":
            with in_process_channel(policy, mode) as channel:  # type: ignore[arg-type]
                yield channel
            return
        with self._socket_dir() as directory:
            host = (
                f"unix://{directory}/policy.sock"
                if self.name == "unix"
                else "127.0.0.1"
            )
            with serving(policy, port=0, host=host, mode=mode) as port:  # type: ignore[arg-type]
                target = host if self.name == "unix" else f"{host}:{port}"
                with grpc.insecure_channel(
                    target, options=channel_options()
                ) as channel:
                    yield channel

    @contextmanager
    def serve_servicers(
        self, *registrations: Callable[[Any], None]
    ) -> Iterator[grpc.Channel]:
        """Serve servicers registered by ``add_*_to_server``-style callables."""
        if self.name == "in_process":
            channel = InProcessChannel()
            for register in registrations:
                register(channel)
            with channel:
                yield channel
            return
        with self._socket_dir() as directory:
            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=2), options=channel_options()
            )
            for register in registrations:
                register(server)
            if self.name == "unix":
                target = f"unix://{directory}/policy.sock"
                server.add_insecure_port(target)
            else:
                target = f"127.0.0.1:{server.add_insecure_port('127.0.0.1:0')}"
            server.start()
            try:
                with grpc.insecure_channel(
                    target, options=channel_options()
                ) as channel:
                    yield channel
            finally:
                server.stop(grace=None)


@pytest.fixture(params=TRANSPORTS)
def transport(request: pytest.FixtureRequest) -> Transport:
    return Transport(request.param)
