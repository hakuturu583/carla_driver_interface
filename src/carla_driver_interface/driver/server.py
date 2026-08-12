# SPDX-License-Identifier: Apache-2.0
"""Serving helpers for the driver process."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager

import grpc

from carla_driver_interface.driver.base import BaseDriver
from carla_driver_interface.driver.service import CarlaEgodriverServicer
from carla_driver_interface.grpc_api import (
    add_EgodriverServiceServicer_to_server,
    channel_options,
)

logger = logging.getLogger(__name__)

__all__ = ["build_server", "run_server", "serving"]


def build_server(
    driver: BaseDriver,
    port: int = 50051,
    host: str = "0.0.0.0",
    max_workers: int = 8,
) -> tuple[grpc.Server, int]:
    """Create (but do not start) a server for ``driver``.

    Returns the server and the bound port, which differs from ``port`` when
    ``port`` is 0 -- useful for tests.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=channel_options(),
    )
    add_EgodriverServiceServicer_to_server(CarlaEgodriverServicer(driver), server)
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
) -> Iterator[int]:
    """Run a driver server for the duration of the block, yielding its port."""
    server, bound_port = build_server(driver, port=port, host=host, max_workers=max_workers)
    server.start()
    logger.info("driver %r serving on %s:%d", driver.name, host, bound_port)
    try:
        yield bound_port
    finally:
        server.stop(grace=1.0).wait()


def run_server(
    driver: BaseDriver,
    port: int = 50051,
    host: str = "0.0.0.0",
    max_workers: int = 8,
) -> None:
    """Serve until the process is interrupted."""
    with serving(driver, port=port, host=host, max_workers=max_workers):
        try:
            # `serving` owns the shutdown; this just parks the main thread.
            threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("shutting down")
