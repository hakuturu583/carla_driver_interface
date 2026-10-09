"""Both ends of alpasim's ``egodriver.EgodriverService`` gRPC contract, without CARLA.

A *policy* -- anything that plans a trajectory from camera frames, ego motion and a
route -- implements :class:`~carla_driver_interface.driver.BaseDriver` and is served
with :func:`~carla_driver_interface.server.run_server` (or
``carla-driver-interface serve``). A *runtime* drives it: autoware_carla_scenario
(https://github.com/autowarefoundation/autoware_carla_scenario, with
``ego.entity=carla_driver``) against CARLA, an upstream alpasim runtime, or
:mod:`carla_driver_interface.testing` without a simulator.

The package is deliberately light -- numpy, scipy, grpcio, protobuf and Pillow -- so a
policy depends on it without pulling in a simulator or a scenario framework, and it runs
on Python 3.10-3.14. The wire contract is vendored from NVlabs/alpasim (see
``proto/README.md`` in the source repository); field numbers and service names are
unchanged, so it interoperates with upstream. The package is versioned on its own, and
the wire contract is its compatibility unit.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("carla-driver-interface")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
