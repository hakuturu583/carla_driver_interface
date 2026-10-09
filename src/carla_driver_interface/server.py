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
from .protocol import add_EgodriverServiceServicer_to_server, channel_options
from .service import EgodriverServicer, ServiceMode

logger = logging.getLogger(__name__)

__all__ = ["build_server", "run_server", "serving"]


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
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=channel_options()
    )
    add_EgodriverServiceServicer_to_server(EgodriverServicer(driver, mode), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    return server, bound_port


@contextmanager
def serving(
    driver: BaseDriver,
    port: int = 50051,
    host: str = "0.0.0.0",
    max_workers: int = 8,
    mode: ServiceMode = "carla",
) -> Iterator[int]:
    """Run a server for ``driver`` for the duration of the block; yields its port."""
    server, bound_port = build_server(
        driver, port=port, host=host, max_workers=max_workers, mode=mode
    )
    server.start()
    logger.info(
        "policy %r serving on %s:%d (%s mode)", driver.name, host, bound_port, mode
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
