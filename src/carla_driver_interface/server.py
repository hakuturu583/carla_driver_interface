"""Serving a policy: build, run, or scope a gRPC server around a :class:`BaseDriver`."""

from __future__ import annotations

import logging
import signal
import threading
from concurrent import futures
from contextlib import contextmanager
from typing import Iterator, Tuple

import grpc

from .driver import BaseDriver
from .protocol import (
    add_CarlaVehicleServiceServicer_to_server,
    add_EgodriverServiceServicer_to_server,
    channel_options,
)
from .service import EgodriverServicer, ServiceMode

logger = logging.getLogger(__name__)

__all__ = ["UNIX_SCHEMES", "bind_address", "build_server", "run_server", "serving"]

#: gRPC address schemes naming a Unix domain socket rather than a TCP host.
UNIX_SCHEMES = ("unix:", "unix-abstract:")


def bind_address(host: str, port: int) -> str:
    """The address :func:`build_server` binds: ``host:port``, or ``host`` itself
    when it names a Unix domain socket (``unix:///run/policy.sock``,
    ``unix:relative.sock``, ``unix-abstract:name``), which has no port."""
    return host if host.startswith(UNIX_SCHEMES) else f"{host}:{port}"


def build_server(
    driver: BaseDriver,
    port: int = 50051,
    host: str = "0.0.0.0",
    max_workers: int = 8,
    mode: ServiceMode = "carla",
) -> Tuple[grpc.Server, int]:
    """Create (but do not start) a server for ``driver``.

    Returns the server and the bound port, which differs from ``port`` when ``port``
    is 0 -- useful for tests. ``mode`` names the runtime on the other end
    (:class:`~carla_driver_interface.service.EgodriverServicer`).

    ``host`` may instead be a Unix domain socket address (``unix:///path``, see
    :func:`bind_address`); ``port`` is then unused and the bound port is 0. A
    client dials the same address.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=channel_options()
    )
    servicer = EgodriverServicer(driver, mode)
    add_EgodriverServiceServicer_to_server(servicer, server)
    # The vehicle command the egodriver contract has no field for.
    add_CarlaVehicleServiceServicer_to_server(servicer, server)
    address = bind_address(host, port)
    try:
        bound_port = server.add_insecure_port(address)
    except RuntimeError:
        bound_port = 0
    if bound_port == 0:
        raise RuntimeError(f"failed to bind {address}")
    # A Unix socket has no port; grpc answers 1 for one that bound.
    return server, 0 if host.startswith(UNIX_SCHEMES) else bound_port


@contextmanager
def serving(
    driver: BaseDriver,
    port: int = 50051,
    host: str = "0.0.0.0",
    max_workers: int = 8,
    mode: ServiceMode = "carla",
) -> Iterator[int]:
    """Run a server for ``driver`` for the duration of the block; yields its port
    (0 on a Unix socket, :func:`build_server`)."""
    server, bound_port = build_server(
        driver, port=port, host=host, max_workers=max_workers, mode=mode
    )
    server.start()
    logger.info(
        "policy %r serving on %s (%s mode)",
        driver.name,
        bind_address(host, bound_port),
        mode,
    )
    try:
        yield bound_port
    finally:
        server.stop(grace=1.0).wait()


def run_server(
    driver: BaseDriver,
    port: int = 50051,
    host: str = "0.0.0.0",
    max_workers: int = 8,
    mode: ServiceMode = "carla",
) -> None:
    """Serve until the process is interrupted or terminated.

    SIGTERM stops the server as Ctrl-C does, so ``docker stop`` (and alpasim's
    teardown of its driver container) closes it cleanly instead of killing it.
    """
    stop = threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    with serving(driver, port=port, host=host, max_workers=max_workers, mode=mode):
        try:
            stop.wait()
        except KeyboardInterrupt:
            pass
        logger.info("shutting down")
