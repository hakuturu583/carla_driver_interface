# SPDX-License-Identifier: Apache-2.0
"""The shared polyline maths.

This module exists because three copies of it had already drifted apart -- one
had an index guard the others lacked, and the three disagreed about what happens
past the end of the line. These tests pin the behaviour that used to differ.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carla_driver_interface import polyline

# An L: 10 m east, then 10 m north. Total length 20 m, one right-angle corner.
ELBOW = np.array(
    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
)


def test_arc_lengths_measures_in_the_xy_plane_only():
    """A route that climbs is not longer for it."""
    climbing = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 100.0]])
    assert polyline.arc_lengths(climbing)[-1] == pytest.approx(5.0)


def test_arc_lengths_of_an_empty_polyline_is_empty():
    assert polyline.arc_lengths(np.zeros((0, 3))).shape == (0,)


def test_sample_interpolates_within_a_segment():
    arc = polyline.arc_lengths(ELBOW)
    assert np.allclose(polyline.sample(ELBOW, arc, 5.0), [5.0, 0.0, 0.0])
    assert np.allclose(polyline.sample(ELBOW, arc, 15.0), [10.0, 5.0, 0.0])


def test_sample_is_vectorised_and_matches_the_scalar_form():
    arc = polyline.arc_lengths(ELBOW)
    queries = np.linspace(0.0, 20.0, 21)
    batch = polyline.sample(ELBOW, arc, queries)
    assert batch.shape == (21, 3)
    for i, d in enumerate(queries):
        assert np.allclose(batch[i], polyline.sample(ELBOW, arc, float(d)))


def test_sample_returns_a_bare_vector_for_a_scalar_query():
    arc = polyline.arc_lengths(ELBOW)
    assert polyline.sample(ELBOW, arc, 5.0).shape == (3,)


def test_sample_clamps_before_the_start():
    """There is no heading to run backwards along, so the start is a wall."""
    arc = polyline.arc_lengths(ELBOW)
    assert np.allclose(polyline.sample(ELBOW, arc, -5.0), ELBOW[0])
    assert np.allclose(polyline.sample(ELBOW, arc, -5.0, extrapolate=True), ELBOW[0])


def test_sample_clamps_past_the_end_by_default():
    arc = polyline.arc_lengths(ELBOW)
    assert np.allclose(polyline.sample(ELBOW, arc, 100.0), ELBOW[-1])


def test_sample_extrapolates_past_the_end_when_asked():
    """The plan builder needs this: a short route must not collapse the plan."""
    arc = polyline.arc_lengths(ELBOW)
    # The final segment heads due north, so 5 m past the end is 5 m further north.
    assert np.allclose(polyline.sample(ELBOW, arc, 25.0, extrapolate=True), [10.0, 15.0, 0.0])


def test_sample_mixes_clamped_and_extrapolated_queries_in_one_call():
    arc = polyline.arc_lengths(ELBOW)
    got = polyline.sample(ELBOW, arc, np.array([5.0, 25.0]), extrapolate=True)
    assert np.allclose(got[0], [5.0, 0.0, 0.0])
    assert np.allclose(got[1], [10.0, 15.0, 0.0])


def test_sample_needs_two_points():
    with pytest.raises(ValueError, match="at least two points"):
        polyline.sample(np.zeros((1, 3)), np.zeros(1), 0.0)


def test_sample_survives_a_zero_length_first_segment():
    """The guard one of the three old copies had and the others did not."""
    doubled = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    arc = polyline.arc_lengths(doubled)
    assert np.allclose(polyline.sample(doubled, arc, 0.0), [0.0, 0.0, 0.0])
    assert np.allclose(polyline.sample(doubled, arc, 4.0), [4.0, 0.0, 0.0])


def test_segment_heading():
    assert polyline.segment_heading(ELBOW[0], ELBOW[1]) == pytest.approx(0.0)
    assert polyline.segment_heading(ELBOW[1], ELBOW[2]) == pytest.approx(math.pi / 2)


def test_max_curvature_of_a_circle_is_one_over_its_radius():
    radius = 25.0
    thetas = np.linspace(0.0, math.pi / 2, 60)
    circle = np.stack(
        [radius * np.sin(thetas), radius * (1 - np.cos(thetas)), np.zeros_like(thetas)], axis=1
    )
    assert polyline.max_curvature(circle) == pytest.approx(1.0 / radius, rel=1e-3)


def test_max_curvature_of_a_straight_line_is_zero():
    straight = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
    assert polyline.max_curvature(straight) == 0.0


def test_max_curvature_needs_three_points():
    assert polyline.max_curvature(np.zeros((2, 3))) == 0.0


def test_max_curvature_ignores_duplicate_points():
    """Repeated vertices make the Menger formula divide by zero."""
    with_duplicates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert polyline.max_curvature(with_duplicates) == 0.0
