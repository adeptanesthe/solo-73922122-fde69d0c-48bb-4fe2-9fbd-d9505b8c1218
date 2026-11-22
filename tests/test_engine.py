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


def _queue_fixture():
    """Three components: [0..3] two labels, [4..5] label-free, [6..7] labels.

    Node 1 (2/3, 1/3) and node 2 (1/3, 2/3) both have uncertainty 1/3, so
    sample-id order decides between them.  Nodes 4 and 5 are unreachable and
    therefore have uncertainty 1.0.  Node 7 is certain (uncertainty 0).
    """

    adjacency = np.zeros((8, 8), dtype=bool)
    for left, right in [(0, 1), (1, 2), (2, 3), (4, 5), (6, 7)]:
        adjacency[left, right] = adjacency[right, left] = True
    components = engine.find_components(adjacency)
    labels = {0: "red", 3: "blue", 6: "red"}
    recommendations = engine.compute_recommendations(adjacency, components, labels)
    return adjacency, components, recommendations, list(range(1, 9))


def test_material_uncertainty_uses_one_minus_highest_probability():
    _, _, recommendations, _ = _queue_fixture()
    assert engine.material_uncertainty(recommendations[1]) == pytest.approx(1 / 3)
    assert engine.material_uncertainty(recommendations[2]) == pytest.approx(1 / 3)
    # No reachable lab point: defined as fully uncertain.
    assert engine.material_uncertainty(recommendations[4]) == 1.0
    # A certain sample contributes no uncertainty.
    assert engine.material_uncertainty(recommendations[7]) == 0.0


def test_lab_queue_covers_each_component_before_filling_by_uncertainty():
    _, components, recommendations, sample_ids = _queue_fixture()

    one = engine.compute_lab_queue(components, recommendations, sample_ids, 1)
    # Only one slot: the most uncertain component's best candidate wins.
    assert [item["sample_id"] for item in one["queue"]] == [5]

    two = engine.compute_lab_queue(components, recommendations, sample_ids, 2)
    # Second slot goes to the next unrepresented component, not to node 4
    # (uncertainty 1.0) in the component already covered.
    assert [item["sample_id"] for item in two["queue"]] == [5, 2]

    three = engine.compute_lab_queue(components, recommendations, sample_ids, 3)
    assert [item["sample_id"] for item in three["queue"]] == [5, 2, 8]
    assert three["covered_component_count"] == 3

    four = engine.compute_lab_queue(components, recommendations, sample_ids, 4)
    # All components covered; the spare slot goes to the most uncertain
    # remaining sample (node 4, uncertainty 1.0).
    assert [item["sample_id"] for item in four["queue"]] == [5, 2, 8, 6]


def test_lab_queue_breaks_uncertainty_ties_by_sample_id():
    _, components, recommendations, sample_ids = _queue_fixture()
    result = engine.compute_lab_queue(components, recommendations, sample_ids, 5)
    queue = [item["sample_id"] for item in result["queue"]]
    # Nodes 2 and 3 tie at uncertainty 1/3; node 2 has the smaller id.
    assert queue.index(2) < queue.index(3)
    assert result["candidate_count"] == 5
    assert len(set(queue)) == len(queue) == 5


def test_lab_queue_never_exceeds_candidate_count():
    _, components, recommendations, sample_ids = _queue_fixture()
    result = engine.compute_lab_queue(components, recommendations, sample_ids, 10)
    assert len(result["queue"]) == 5
    assert result["component_count"] == 3


def test_lab_queue_is_deterministic_for_fixed_input():
    _, components, recommendations, sample_ids = _queue_fixture()
    first = engine.compute_lab_queue(components, recommendations, sample_ids, 4)
    second = engine.compute_lab_queue(components, recommendations, sample_ids, 4)
    assert first == second
