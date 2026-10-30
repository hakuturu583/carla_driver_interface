"""A policy fed T4Scenes: :class:`T4SceneDriver`, a :class:`BaseDriver` like any other.

tier4/e2e-devkit wraps a trained model as a *T4 driver*
(``t4_e2e_devkit.driver.AbstractT4Driver``): a ``T4Scene`` goes in, an
ego-frame plan comes out, and the model is fed exactly the window it was
trained on.  :class:`T4SceneDriver` puts one behind the egodriver contract.
The runtime sends the rollout so far as a scene with every drive
(``CarlaRendererData.t4_scene``, ``driver.t4_scene``), and the map it
references once, as the map set's ``t4`` format; this class decodes both, hands
the scene to the T4 driver under the session's id, and returns its plan as a
:class:`DriveResult` like any other policy's.

Nothing else changes: the session still records every observation, so a
subclass can read ``ctx.session`` beside the scene, and a policy that does not
read scenes keeps subclassing :class:`BaseDriver` itself.

Needs the devkit (``t4-e2e-devkit``), imported when the first scene is decoded:
this package does not depend on it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from .driver import BaseDriver, DriveContext, DriveResult, SessionState, TurnIndicator
from .geometry import Pose, Trajectory

if TYPE_CHECKING:
    from .hdmap import MapFiles

__all__ = ["T4_MAP_FORMAT", "T4SceneDriver"]

logger = logging.getLogger(__name__)

#: The map set format holding the scene's map resource: a serialized
#: ``t4_e2e_devkit`` ``MapResource`` (Lanelet2 OSM in Autoware's ``map`` frame).
T4_MAP_FORMAT = "t4"


class T4SceneDriver(BaseDriver):
    """Drives with a devkit T4 driver over the scene the runtime sends.

    Args:
        scene_driver: The model, as a ``t4_e2e_devkit.driver.AbstractT4Driver``.
        map_dir: Your copy of the runtime's ``driver.map_dir``, where the scene's
            map resource is (the ``t4`` format); ``None`` for a model that reads
            no map.
        name: Reported as the policy's name; the T4 driver's own by default.

    The plan the T4 driver returns is in the ego frame of the scene's current
    row, the rig at ``time_now_us``; pose ``i`` is ``(i + 1)`` sampling
    intervals later.  Its ``extras`` may carry ``turn_indicator`` (an
    :class:`TurnIndicator` value) and ``debug_scalars`` (a ``str -> float``
    mapping), which the result passes on.
    """

    map_format: str = T4_MAP_FORMAT

    def __init__(
        self,
        scene_driver: Any,
        *,
        map_dir: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.scene_driver = scene_driver
        self.map_dir = map_dir
        self.name = name if name is not None else str(scene_driver.name())
        #: The map resources registered so far, one store for every map.
        self._resources: Any = None
        #: Map sets whose resource is registered, by map id.
        self._registered: set[str] = set()
        self._lock = threading.Lock()

    # -- sessions -------------------------------------------------------------

    def on_session_start(self, session: SessionState) -> None:
        self.scene_driver.start_session(session.uuid)

    def on_session_close(self, session: SessionState) -> None:
        self.scene_driver.close_session(session.uuid)

    # -- the decision -----------------------------------------------------------

    def drive(self, ctx: DriveContext) -> DriveResult:
        output = self.scene_driver.drive(self.scene(ctx), session_id=ctx.session.uuid)
        return self.result(output, ctx)

    def scene(self, ctx: DriveContext) -> Any:
        """The step's ``T4Scene``, its map resolved.

        Raises:
            RuntimeError: when the runtime sent no scene; it sends one only when
                asked to (``driver.t4_scene``).
        """
        payload = None if ctx.renderer_data is None else ctx.renderer_data.t4_scene
        if not payload:
            raise RuntimeError(
                f"{self.name} plans from T4Scenes, and this drive carries none: run the "
                "runtime with driver.t4_scene on"
            )
        from t4_e2e_devkit.serialization.protobuf import from_proto, t4_scene_pb2

        message = t4_scene_pb2.T4Scene.FromString(payload)
        return from_proto(message, map_resources=self._map_resources(ctx.map))

    def result(self, output: Any, ctx: DriveContext) -> DriveResult:
        """A T4 driver's ``DriverOutput`` as the egodriver's result, in the rig frame."""
        trajectory = output.trajectory
        poses = np.asarray(trajectory.poses, dtype=np.float64).reshape(-1, 3)
        step_us = int(
            round(float(trajectory.trajectory_sampling.interval_length) * 1e6)
        )
        plan = Trajectory.empty()
        for index, (x, y, yaw) in enumerate(poses, start=1):
            plan.append(
                ctx.time_now_us + index * step_us, Pose.from_xyz_yaw(x, y, 0.0, yaw)
            )
        extras: Dict[str, Any] = dict(getattr(output, "extras", None) or {})
        indicator = extras.get("turn_indicator")
        return DriveResult(
            trajectory_in_rig=plan,
            turn_indicator=(
                TurnIndicator.NO_COMMAND
                if indicator is None
                else TurnIndicator(int(indicator))
            ),
            debug_scalars={
                str(key): float(value)
                for key, value in dict(extras.get("debug_scalars") or {}).items()
            },
        )

    def _map_resources(self, map_files: Optional["MapFiles"]) -> Any:
        """The store holding the step's map resource; ``None`` without a map set.

        Each set's resource is read and verified once (it does not change under
        its id); the store keeps every map registered so far.
        """
        if map_files is None:
            return None
        from t4_e2e_devkit.serialization.protobuf import MapResourceStore, t4_scene_pb2

        with self._lock:
            if self._resources is None:
                # Room for every map a server is likely to meet; an evicted map
                # would be read again only if _registered forgot it too.
                self._resources = MapResourceStore(max_maps=64)
            if map_files.map_id not in self._registered:
                resource = t4_scene_pb2.MapResource.FromString(
                    Path(map_files.path).read_bytes()
                )
                self._resources.register(resource)
                self._registered.add(map_files.map_id)
                logger.info("registered the T4 map of %s", map_files.map_id)
            return self._resources
