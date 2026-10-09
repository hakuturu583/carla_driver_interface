"""Reference policies.

Deliberately dependency-free (numpy only), so the closed loop can be exercised
without a model checkpoint or a GPU.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Type

from carla_driver_interface.driver import BaseDriver
from carla_driver_interface.policies.constant_speed import ConstantSpeedPolicy
from carla_driver_interface.policies.route_follower import RouteFollowerPolicy

#: Name -> class, for ``carla-driver-interface serve --policy``.
POLICY_REGISTRY: Dict[str, Type[BaseDriver]] = {
    ConstantSpeedPolicy.name: ConstantSpeedPolicy,
    RouteFollowerPolicy.name: RouteFollowerPolicy,
}


def load_policy(spec: str, **kwargs: Any) -> BaseDriver:
    """Build the policy *spec* names, with *kwargs*.

    *spec* is a name in :data:`POLICY_REGISTRY` (``route_follower``) or
    ``package.module:attribute`` -- a :class:`BaseDriver` subclass, or any callable
    returning one -- so a policy of your own is named the way a reference one is.

    Raises:
        ValueError: *spec* names nothing that builds a :class:`BaseDriver`.
    """
    if spec in POLICY_REGISTRY:
        factory: Any = POLICY_REGISTRY[spec]
    elif ":" in spec:
        module, _, attribute = spec.partition(":")
        try:
            factory = getattr(importlib.import_module(module), attribute)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot load policy {spec!r}: {exc}") from exc
    else:
        raise ValueError(
            f"unknown policy {spec!r}: expected one of {sorted(POLICY_REGISTRY)} "
            "or 'package.module:Class'"
        )
    try:
        policy = factory(**kwargs)
    except TypeError as exc:
        raise ValueError(f"cannot build policy {spec!r}: {exc}") from exc
    if not isinstance(policy, BaseDriver):
        raise ValueError(
            f"policy {spec!r} built a {type(policy).__name__}, not a BaseDriver"
        )
    return policy


__all__ = [
    "POLICY_REGISTRY",
    "ConstantSpeedPolicy",
    "RouteFollowerPolicy",
    "load_policy",
]
