# SPDX-License-Identifier: Apache-2.0
"""Route publishing, matching what alpasim's ``RouteGenerator`` does.

alpasim projects the ego onto the map, takes the stretch of route ahead, and
sends it to the driver **in the rig frame at the current timestamp**
(``events/policy.py``).  :class:`RouteProvider` does the same for a route
polyline expressed in the ``local`` frame, and holds no CARLA references so it
can be tested directly.
"""

from __future__ import annotations

import numpy as np

from carla_driver_interface.geometry import Pose

__all__ = ["RouteProvider"]


class RouteProvider:
    """Slices a global route into the window the driver sees each step."""

    def __init__(
        self,
        route_in_local: np.ndarray,
        horizon_m: float = 80.0,
        resolution_m: float = 2.0,
    ) -> None:
        route = np.asarray(route_in_local, dtype=np.float64).reshape(-1, 3)
        if len(route) < 2:
            raise ValueError("a route needs at least two points")
        self._route = route
        self._arc = _arc_lengths(route)
        self.horizon_m = horizon_m
        self.resolution_m = resolution_m
        #: Monotonic progress along the route, so the ego cannot be matched back
        #: onto an earlier lap of a loop or onto the opposite side of a U-turn.
        self._progress_m = 0.0

    @property
    def total_length_m(self) -> float:
        return float(self._arc[-1])

    @property
    def progress_m(self) -> float:
        return self._progress_m

    @property
    def completion(self) -> float:
        """Fraction of the route driven so far, in ``[0, 1]``."""
        if self.total_length_m <= 0.0:
            return 1.0
        return min(1.0, self._progress_m / self.total_length_m)

    @property
    def remaining_m(self) -> float:
        """Route length still ahead of the ego."""
        return max(0.0, self.total_length_m - self._progress_m)

    def lateral_error_m(self, pose_local_to_rig: Pose) -> float:
        """Signed offset of the ego from the route centreline.

        Positive means the ego is to the left of the route. This is the honest
        tracking metric: the driver's plan is anchored on the ego, so measuring
        the ego against *that* is structurally zero.
        """
        nearest = self._sample(self._advance_progress(pose_local_to_rig.position))
        return float(pose_local_to_rig.inverse().transform_points(nearest)[0][1])

    def waypoints_in_rig(self, pose_local_to_rig: Pose) -> np.ndarray:
        """The route ahead of ``pose_local_to_rig``, resampled, in the rig frame.

        Advances the internal progress marker as a side effect, so call this
        once per policy step.
        """
        self._progress_m = self._advance_progress(pose_local_to_rig.position)

        end = min(self.total_length_m, self._progress_m + self.horizon_m)
        distances = np.arange(self._progress_m, end, self.resolution_m)
        # Always finish on the exact endpoint. Without it the last stride is
        # truncated, the ego can never reach 100% completion, and the rollout
        # runs off the end of the route into undefined behaviour.
        distances = np.append(distances, end)
        if len(distances) < 2:
            distances = np.array([max(0.0, end - self.resolution_m), end])

        points_local = np.stack([self._sample(d) for d in distances])
        return pose_local_to_rig.inverse().transform_points(points_local)

    # -- internals ---------------------------------------------------------

    def _advance_progress(self, position_local: np.ndarray) -> float:
        """Project the ego onto the route, searching forward from the last match.

        The search window is generous enough to absorb a step's worth of motion
        plus a large tracking error, but bounded so a loop route never matches
        backwards.
        """
        search_end = min(self.total_length_m, self._progress_m + max(self.horizon_m, 30.0))
        samples = np.arange(self._progress_m, search_end, self.resolution_m)
        # Include the endpoint so progress can actually reach the route's end.
        samples = np.append(samples, search_end)
        if len(samples) == 0:
            return self._progress_m

        points = np.stack([self._sample(d) for d in samples])
        squared = np.sum((points[:, :2] - position_local[:2]) ** 2, axis=1)
        return float(samples[int(np.argmin(squared))])

    def _sample(self, distance: float) -> np.ndarray:
        distance = float(np.clip(distance, 0.0, self.total_length_m))
        if distance >= self.total_length_m:
            return self._route[-1]
        idx = max(1, int(np.searchsorted(self._arc, distance)))
        p0, p1 = self._route[idx - 1], self._route[idx]
        span = self._arc[idx] - self._arc[idx - 1]
        alpha = 0.0 if span <= 0.0 else (distance - self._arc[idx - 1]) / span
        return p0 + alpha * (p1 - p0)


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(deltas)])
