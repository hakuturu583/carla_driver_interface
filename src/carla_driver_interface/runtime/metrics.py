# SPDX-License-Identifier: Apache-2.0
"""Rollout scoring, reported in alpasim's own result types.

``runtime.proto``'s ``SimulationReturn.RolloutReturn`` carries both per-timestep
series and aggregate scalars, so there is no reason to invent a result type --
reusing it means an alpasim-side consumer can read a CARLA rollout unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from carla_driver_interface.geometry import Pose, Trajectory
from carla_driver_interface.grpc_api import (
    RolloutErrorCode,
    RolloutSpec,
    SimulationReturn,
    TimeAggregation,
)

__all__ = ["MetricsCollector"]

_TimestepMetric = SimulationReturn.TimestepMetric


@dataclass
class _Series:
    """One named per-timestep series."""

    aggregation: TimeAggregation
    timestamps_us: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(self, timestamp_us: int, value: float) -> None:
        self.timestamps_us.append(int(timestamp_us))
        self.values.append(float(value))


class MetricsCollector:
    """Accumulates per-step measurements and reduces them at the end."""

    def __init__(self) -> None:
        self._series: dict[str, _Series] = {}
        self._distance_m = 0.0
        self._previous_position: np.ndarray | None = None
        #: Plans keyed by the step that produced them, for plan-deviation.
        self._pending_plans: list[Trajectory] = []
        self._plan_deviations: list[float] = []

    # -- recording ---------------------------------------------------------

    def record_step(
        self,
        timestamp_us: int,
        pose_local_to_rig: Pose,
        speed_mps: float,
        target_speed_mps: float,
        route_lateral_error_m: float,
        lookahead_lateral_offset_m: float,
        route_completion: float,
        drive_latency_s: float,
    ) -> None:
        position = pose_local_to_rig.position
        if self._previous_position is not None:
            self._distance_m += float(np.linalg.norm(position[:2] - self._previous_position[:2]))
        self._previous_position = position

        self._add("speed_mps", TimeAggregation.TIME_AGGREGATION_MEAN, timestamp_us, speed_mps)
        self._add(
            "target_speed_mps",
            TimeAggregation.TIME_AGGREGATION_MEAN,
            timestamp_us,
            target_speed_mps,
        )
        # How far the ego actually is from the route it was asked to follow --
        # the real tracking error, measured by the runtime rather than inferred
        # from the driver's own (ego-anchored) plan.
        self._add(
            "route_lateral_error_m",
            TimeAggregation.TIME_AGGREGATION_MAX,
            timestamp_us,
            abs(route_lateral_error_m),
        )
        self._add(
            "lookahead_lateral_offset_m",
            TimeAggregation.TIME_AGGREGATION_MAX,
            timestamp_us,
            abs(lookahead_lateral_offset_m),
        )
        self._add(
            "route_completion",
            TimeAggregation.TIME_AGGREGATION_LAST,
            timestamp_us,
            route_completion,
        )
        self._add(
            "drive_latency_s",
            TimeAggregation.TIME_AGGREGATION_MEAN,
            timestamp_us,
            drive_latency_s,
        )

        self._score_pending_plans(timestamp_us, position)

    def record_plan(self, plan_in_local: Trajectory) -> None:
        """Remember a plan so later steps can measure how far the ego strayed.

        This mirrors alpasim's ``plan_deviation`` scorer: compare each plan
        against where the ego actually went at the same timestamps.
        """
        if len(plan_in_local) >= 2:
            self._pending_plans.append(plan_in_local)

    def _score_pending_plans(self, timestamp_us: int, position: np.ndarray) -> None:
        still_pending = []
        for plan in self._pending_plans:
            if timestamp_us > plan.timestamps_us[-1]:
                continue  # fully consumed
            still_pending.append(plan)
            if timestamp_us < plan.timestamps_us[0]:
                continue
            predicted = plan.interpolate(timestamp_us).position
            self._plan_deviations.append(float(np.linalg.norm(predicted[:2] - position[:2])))
        self._pending_plans = still_pending

    def _add(
        self, name: str, aggregation: TimeAggregation, timestamp_us: int, value: float
    ) -> None:
        series = self._series.get(name)
        if series is None:
            series = self._series[name] = _Series(aggregation=aggregation)
        series.add(timestamp_us, value)

    # -- reporting ---------------------------------------------------------

    def build_return(
        self,
        rollout_uuid: str,
        scenario_id: str,
        success: bool,
        error: str,
        collisions: int,
        lane_invasions: int,
        route_completion: float,
        error_code: RolloutErrorCode,
    ) -> SimulationReturn.RolloutReturn:
        timestep_metrics = [
            _TimestepMetric(
                name=name,
                timestamps_us=series.timestamps_us,
                values=series.values,
                valid=[True] * len(series.values),
                time_aggregation=series.aggregation,
            )
            for name, series in self._series.items()
        ]

        speed_series = self._series.get("speed_mps")
        aggregated = {
            "collisions": float(collisions),
            "lane_invasions": float(lane_invasions),
            "distance_travelled_m": self._distance_m,
            "route_completion": route_completion,
            "steps": float(len(speed_series.values)) if speed_series else 0.0,
        }
        for name, series in self._series.items():
            if series.values:
                aggregated[name] = _reduce(series)
        if self._plan_deviations:
            aggregated["plan_deviation_mean_m"] = float(np.mean(self._plan_deviations))
            aggregated["plan_deviation_max_m"] = float(np.max(self._plan_deviations))

        return SimulationReturn.RolloutReturn(
            rollout_spec=RolloutSpec(scenario_id=scenario_id, nr_rollouts=1),
            success=success,
            rollout_uuid=rollout_uuid,
            error=error,
            timestep_metrics=timestep_metrics,
            aggregated_metrics=aggregated,
            error_code=error_code,
        )


def _reduce(series: _Series) -> float:
    values = np.asarray(series.values, dtype=np.float64)
    match series.aggregation:
        case TimeAggregation.TIME_AGGREGATION_MEDIAN:
            return float(np.median(values))
        case TimeAggregation.TIME_AGGREGATION_MAX:
            return float(np.max(values))
        case TimeAggregation.TIME_AGGREGATION_MIN:
            return float(np.min(values))
        case TimeAggregation.TIME_AGGREGATION_LAST:
            return float(values[-1])
        case _:
            return float(np.mean(values))
