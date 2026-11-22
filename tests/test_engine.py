"""Tests for graph construction and first-hit probability solving."""

from __future__ import annotations

import numpy as np
import pytest

from app import engine


def _path_adjacency() -> np.ndarray:
    adjacency = np.zeros((4, 4), dtype=bool)
    for left, right in [(0, 2), (2, 3), (3, 1)]:
        adjacency[left, right] = adjacency[right, left] = True
    return adjacency


def test_first_hit_probabilities_on_path():
    adjacency = _path_adjacency()
    components = engine.find_components(adjacency)
    assert len(components) == 1
    recommendations = engine.compute_recommendations(
        adjacency, components, {0: "red", 1: "blue"}
    )

    assert set(recommendations) == {2, 3}
    assert recommendations[2].probabilities["red"] == pytest.approx(2 / 3)
    assert recommendations[2].probabilities["blue"] == pytest.approx(1 / 3)
    assert recommendations[3].probabilities["red"] == pytest.approx(1 / 3)
    # Below 0.8, so both samples are pending rather than binned.
    assert recommendations[2].status == engine.STATUS_PENDING_LAB
    assert recommendations[2].reason == engine.REASON_BELOW_THRESHOLD


def test_threshold_is_inclusive_on_raw_euclidean_distance():
    vectors = np.asarray([[0.0, 1.0], [3.0, 4.0]])
    expected = float(np.linalg.norm(np.asarray([0.0, 1.0]) - np.asarray([3.0, 4.0])))
    adjacency = engine.build_adjacency(vectors, expected)
    assert adjacency[0, 1] and adjacency[1, 0]
    assert not np.any(np.diag(adjacency))

    adjacency_stricter = engine.build_adjacency(vectors, expected - 1e-6)
    assert not adjacency_stricter[0, 1]


def test_components_split_when_beyond_threshold():
    # Nodes 0 and 1 are 1.0 apart and connect; node 2 is further from both.
    vectors = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    adjacency = engine.build_adjacency(vectors, 1.0)
    components = engine.find_components(adjacency)
    sizes = sorted(len(component) for component in components)
    assert sizes == [1, 2]


def test_unlabelled_component_is_pending_without_reachable_lab():
    vectors = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    adjacency = engine.build_adjacency(vectors, 1.0)
    components = engine.find_components(adjacency)
    recommendations = engine.compute_recommendations(adjacency, components, {2: "red"})
    assert recommendations[0].status == engine.STATUS_PENDING_LAB
    assert recommendations[0].reason == engine.REASON_NO_REACHABLE_LAB
    assert recommendations[0].probabilities == {}
    # Node 1 is in the same label-free component as node 0.
    assert recommendations[1].reason == engine.REASON_NO_REACHABLE_LAB
    # The labelled node itself receives no recommendation.
    assert 2 not in recommendations


def test_non_finite_vector_is_rejected():
    with pytest.raises(engine.ComputationError):
        engine.build_adjacency(np.asarray([[1.0, np.nan]]), 0.5)


def test_probability_at_threshold_counts_as_binnable():
    # A 4:1 star gives the centre exactly 0.8; the boundary is inclusive.
    adjacency = np.zeros((6, 6), dtype=bool)
    for node in range(1, 6):
        adjacency[0, node] = adjacency[node, 0] = True
    recommendations = engine.compute_recommendations(
        adjacency,
        engine.find_components(adjacency),
        {1: "red", 2: "red", 3: "red", 4: "red", 5: "blue"},
    )
    assert recommendations[0].material == "red"
    assert recommendations[0].probabilities["red"] == pytest.approx(0.8)
    assert recommendations[0].status == engine.STATUS_RECOMMEND_BIN


def test_symmetric_material_tie_stays_pending():
    # A path red -- u -- blue gives u an exact 0.5/0.5 split, which must never
    # be reported as a bin recommendation.
    adjacency = np.zeros((3, 3), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    recommendations = engine.compute_recommendations(
        adjacency, engine.find_components(adjacency), {0: "zeta", 2: "alpha"}
    )
    item = recommendations[1]
    assert item.status == engine.STATUS_PENDING_LAB
    assert item.material is None
    assert item.probabilities["zeta"] == pytest.approx(0.5)
    assert item.probabilities["alpha"] == pytest.approx(0.5)
