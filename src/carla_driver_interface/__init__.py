# SPDX-License-Identifier: Apache-2.0
"""alpasim-compatible egodriver module plus a CARLA closed-loop runtime.

The driver half speaks ``egodriver.EgodriverService`` exactly as
`NVlabs/alpasim <https://github.com/NVlabs/alpasim>`_ defines it, by inheriting
the upstream generated servicer.  The runtime half replaces alpasim's sensorsim,
controller, physics and traffic services with CARLA.  ``docs/COMPATIBILITY.md``
and :mod:`carla_driver_interface.compat` enumerate where the two differ.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

#: alpasim commit that the ``alpasim-grpc`` dependency is pinned to in
#: ``pyproject.toml``.  Kept here so the compat report can state it without
#: parsing package metadata.
ALPASIM_GRPC_REV = "68709245a5dc0f2eda4f8cb2c3aa8cbdfa913043"

try:
    __version__ = _pkg_version("carla-driver-interface")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["ALPASIM_GRPC_REV", "__version__"]
