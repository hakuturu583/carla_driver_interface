# SPDX-License-Identifier: Apache-2.0
"""Serving helpers for the driver process."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager

import grpc

from carla_driver_interface.driver.base import BaseDriver
from carla_driver_interface.driver.service import CarlaEgodriverServicer
from carla_driver_interface.grpc_api import add_EgodriverServiceServicer_to_server

logger = logging.getLogger(__name__)

__all__ = ["build_server", "run_server", "serving"]

#: alpasim submits full camera frames as single unary messages, so the default
#: 4 MiB inbound limit is too small for anything above roughly 1080p PNG.
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


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
        options=[
            ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
        ],
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
    server, bound_port = build_server(driver, port=port, host=host, max_workers=max_workers)
    server.start()
    logger.info("driver %r serving on %s:%d", driver.name, host, bound_port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.stop(grace=2.0).wait()
