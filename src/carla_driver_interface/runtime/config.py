# SPDX-License-Identifier: Apache-2.0
"""Configuration for a CARLA rollout."""

from __future__ import annotations

from dataclasses import dataclass, field

from carla_driver_interface.grpc_api import ImageFormat
from carla_driver_interface.runtime.control import ControlConfig
from carla_driver_interface.runtime.images import validate_image_format

__all__ = ["CameraConfig", "RuntimeConfig", "ScenarioSpec", "default_camera_rig"]


@dataclass(frozen=True)
class CameraConfig:
    """One ``sensor.camera.rgb`` mounted on the ego.

    ``x/y/z`` and the angles are the CARLA-side mount transform *relative to the
    vehicle actor*, in CARLA's own left-handed convention and degrees -- i.e.
    exactly what you would pass to ``carla.Transform``.  The conversion into the
    rig frame happens in :mod:`carla_driver_interface.runtime.conversions`, so
    these numbers can be copied straight out of existing CARLA configs.
    """

    logical_id: str
    width: int = 960
    height: int = 604
    fov_deg: float = 120.0
    x: float = 1.5
    y: float = 0.0
    z: float = 1.6
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0


def default_camera_rig() -> list[CameraConfig]:
    """A single forward camera.

    The logical id follows the alpasim/DRIVE naming so drivers written against
    upstream rig configs find the camera they expect.
    """
    return [CameraConfig(logical_id="camera_front_wide_120fov")]


@dataclass(frozen=True)
class ScenarioSpec:
    """What to simulate."""

    map_name: str = "Town10HD_Opt"
    name: str = "default"
    #: Index into the map's spawn points; ``None`` picks one with the seed.
    spawn_point_index: int | None = None
    #: Background vehicles handed to the CARLA TrafficManager.
    num_background_vehicles: int = 30
    #: CARLA weather preset name, e.g. ``"ClearNoon"``. ``None`` keeps the map default.
    weather_preset: str | None = None

    @property
    def scene_id(self) -> str:
        """The ``scene_id`` reported to the driver.

        alpasim uses a recording UUID here; there are no recordings under CARLA,
        so a map/scenario pair takes its place. See ``docs/COMPATIBILITY.md``.
        """
        return f"{self.map_name}:{self.name}"


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything the runtime needs that is not the scenario itself."""

    # -- connection --
    carla_host: str = "localhost"
    carla_port: int = 2000
    carla_timeout_s: float = 20.0
    traffic_manager_port: int = 8000

    # -- driver --
    driver_address: str = "localhost:50051"
    driver_timeout_s: float = 60.0

    # -- timing --
    #: Simulator tick period. CARLA needs synchronous mode for a reproducible
    #: closed loop.
    fixed_delta_s: float = 0.05
    #: How often the driver is queried. Must be a multiple of ``fixed_delta_s``.
    policy_timestep_s: float = 0.1
    max_steps: int = 400
    #: Ticks to run before the first driver query, letting physics and sensors
    #: settle. alpasim calls the equivalent window the priming phase.
    warmup_ticks: int = 10

    # -- vehicle --
    ego_blueprint: str = "vehicle.tesla.model3"
    #: Signed offset from the CARLA actor origin to the rig origin (rear axle
    #: centre) along the body x axis, in metres -- negative for a normal car.
    #: ``None`` derives it from the wheel physics control, which is worth
    #: overriding: in CARLA 0.9.x ``WheelPhysicsControl.position`` is reported in
    #: **world centimetres**, so the derivation is easy to get wrong on a map
    #: with an unusual origin. The value actually used is always logged.
    rear_axle_offset_m: float | None = None

    # -- sensors --
    cameras: list[CameraConfig] = field(default_factory=default_camera_rig)
    image_format: int = ImageFormat.JPEG
    image_quality: int = 90

    # -- route --
    #: How far ahead of the ego the route is published, in metres.
    route_horizon_m: float = 80.0
    #: Spacing of the published route waypoints, in metres.
    route_resolution_m: float = 2.0

    # -- traffic lights --
    #: How far ahead of the ego a traffic light is looked for, in metres.
    #:
    #: CARLA's own answer to "does a light govern me" is
    #: ``Actor.is_at_traffic_light()``, which is true only inside the light's
    #: trigger volume -- and those are thin. Measured across all fifteen lights
    #: of Town10HD_Opt, the volume reaches a median of 0.55 m back from the
    #: stop line and 2.38 m at the most, so a policy relying on it hears about
    #: a red light once it is already at the line. Stopping from 25 km/h takes
    #: about six metres, so it cannot; the overrun that follows belongs to this
    #: interface rather than to the policy being evaluated.
    #:
    #: Walking the lane graph forward instead reports the light from where a
    #: driver would have seen it. 0 restores the trigger-volume behaviour.
    traffic_light_sight_distance_m: float = 60.0

    # -- observations --
    #: alpasim sends the recorded ground-truth trajectory; there is no recording
    #: under CARLA, so this is off by default. Turning it on sends the route
    #: reference instead, which is *not* the same thing -- see
    #: ``docs/COMPATIBILITY.md``.
    send_ground_truth: bool = False
    #: Standard deviation of the Gaussian egomotion noise, in metres and
    #: radians. 0 makes ``rig_est`` identical to ``rig``.
    egomotion_position_noise_m: float = 0.0
    egomotion_yaw_noise_rad: float = 0.0
    #: Include other actors in the ``CarlaRendererData`` extension payload.
    send_actor_ground_truth: bool = True

    # -- control --
    control: ControlConfig = field(default_factory=ControlConfig)

    # -- misc --
    seed: int = 0
    #: Added to every timestamp, for placing a rollout on an absolute timeline.
    epoch_offset_us: int = 0

    def __post_init__(self) -> None:
        validate_image_format(self.image_format)
        if self.fixed_delta_s <= 0.0:
            raise ValueError("fixed_delta_s must be positive")
        ratio = self.policy_timestep_s / self.fixed_delta_s
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ValueError(
                f"policy_timestep_s ({self.policy_timestep_s}) must be a positive "
                f"multiple of fixed_delta_s ({self.fixed_delta_s})"
            )

    @property
    def ticks_per_policy_step(self) -> int:
        return int(round(self.policy_timestep_s / self.fixed_delta_s))

    @property
    def policy_timestep_us(self) -> int:
        return int(round(self.policy_timestep_s * 1e6))
