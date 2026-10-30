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
    mapping), which the result passes on; the servicer drops a value of
    either it cannot send, keeping the plan.

    One drive at a time: a T4 driver holds a model and per-session state that
    concurrent drives would share.
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
        #: The map resources registered, one store for every map.
        self._store: Any = None
        #: Each map set's resource, read once, by map id.
        self._read: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # -- sessions -------------------------------------------------------------

    def on_session_start(self, session: SessionState) -> None:
        self.scene_driver.start_session(session.uuid)

    def on_session_close(self, session: SessionState) -> None:
        self.scene_driver.close_session(session.uuid)

    # -- the decision -----------------------------------------------------------

    def drive(self, ctx: DriveContext) -> DriveResult:
        with self._lock:
            output = self.scene_driver.drive(
                self.scene(ctx), session_id=ctx.session.uuid
            )
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
        store = self._map_store(ctx.map)
        try:
            return from_proto(message, map_resources=store)
        except KeyError as error:
            # The store keeps the most recent maps; one it let go is
            # registered again from the copy read.
            if ctx.map is None or "not registered" not in str(error):
                raise
            store.register(self._read[ctx.map.map_id])
            return from_proto(message, map_resources=store)

    def result(self, output: Any, ctx: DriveContext) -> DriveResult:
        """A T4 driver's ``DriverOutput`` as the egodriver's result, in the rig frame."""
        trajectory = output.trajectory
        poses = np.asarray(trajectory.poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(
                f"{self.name} returned poses of shape {poses.shape}, not (N, 3) x, y, heading"
            )
        step_us = int(
            round(float(trajectory.trajectory_sampling.interval_length) * 1e6)
        )
        if step_us <= 0:
            raise ValueError(
                f"{self.name} returned a plan sampled every {step_us} us; "
                "its poses would all be at the anchor"
            )
        plan = Trajectory.empty()
        for index, (x, y, yaw) in enumerate(poses, start=1):
            plan.append(
                ctx.time_now_us + index * step_us, Pose.from_xyz_yaw(x, y, 0.0, yaw)
            )
        extras = output.extras or {}
        return DriveResult(
            trajectory_in_rig=plan,
            turn_indicator=extras.get("turn_indicator", TurnIndicator.NO_COMMAND),
            debug_scalars=dict(extras.get("debug_scalars") or {}),
        )

    def _map_store(self, map_files: Optional["MapFiles"]) -> Any:
        """The store holding the step's map resource; ``None`` without a map set.

        Each set's resource is read and registered once: it does not change
        under its id.
        """
        if map_files is None:
            return None
        from t4_e2e_devkit.serialization.protobuf import MapResourceStore, t4_scene_pb2

        if self._store is None:
            # Room for every map a server is likely to meet.
            self._store = MapResourceStore(max_maps=64)
        if map_files.map_id not in self._read:
            resource = t4_scene_pb2.MapResource.FromString(
                Path(map_files.path).read_bytes()
            )
            self._store.register(resource)
            self._read[map_files.map_id] = resource
            logger.info("registered the T4 map of %s", map_files.map_id)
        return self._store
