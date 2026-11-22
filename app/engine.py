"""Random-walk first-hit probability engine (NumPy).

The graph is fixed for a batch: nodes are fabric samples, and an undirected
edge exists when the Euclidean distance between two normalised spectra is at
most the batch threshold.  A random walk starts at an unlabelled sample and,
at every step, chooses one of its neighbours uniformly at random.  Labelled
samples are absorbing; the first labelled sample reached determines the
material.  For each unlabelled sample we solve the harmonic equations for the
probability of being absorbed by each material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

BIN_PROBABILITY = 0.8
PROBABILITY_TOLERANCE = 1e-9
DISTANCE_TOLERANCE = 1e-12
RESIDUAL_TOLERANCE = 1e-7
STATE_TOLERANCE = 1e-6

STATUS_RECOMMEND_BIN = "recommend_bin"
STATUS_PENDING_LAB = "pending_lab"
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_NO_REACHABLE_LAB = "no_reachable_lab_point"


class ComputationError(RuntimeError):
    """Raised when the numerical system cannot be solved reliably."""


@dataclass(frozen=True)
class Recommendation:
    status: str
    material: str | None
    probabilities: dict[str, float]
    reason: str | None

    def as_dict(self) -> dict:
        def _probability(value: float) -> float:
            rounded = round(float(value), 12)
            return 0.0 if rounded == 0.0 else rounded

        return {
            "status": self.status,
            "material": self.material,
            "probabilities": {
                material: _probability(value)
                for material, value in self.probabilities.items()
            },
            "reason": self.reason,
        }


def build_adjacency(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean adjacency matrix using ``distance <= threshold``.

    Spectra are expected to be normalised by the caller; distances are plain
    Euclidean distances on the submitted vectors.
    """

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ComputationError("至少需要一份布样才能构图")
    if not np.all(np.isfinite(vectors)):
        raise ComputationError("光谱向量必须为有限数值")

    count = vectors.shape[0]
    distances = np.empty((count, count), dtype=np.float64)
    for row in range(count):
        difference = vectors - vectors[row]
        distances[row] = np.sqrt(np.einsum("ij,ij->i", difference, difference))
    np.fill_diagonal(distances, 0.0)

    tolerance = DISTANCE_TOLERANCE * max(1.0, abs(float(threshold)))
    adjacency = distances <= float(threshold) + tolerance
    np.fill_diagonal(adjacency, False)
    return adjacency


def find_components(adjacency: np.ndarray) -> list[np.ndarray]:
    """Return connected components as arrays of node indices."""

    count = int(adjacency.shape[0])
    visited = np.zeros(count, dtype=bool)
    components: list[np.ndarray] = []

    for start in range(count):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        nodes: list[int] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            neighbours = np.flatnonzero(adjacency[node])
            for neighbour in neighbours:
                neighbour = int(neighbour)
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        components.append(np.asarray(sorted(nodes), dtype=np.int64))

    return components


def _solve_component(
    adjacency: np.ndarray,
    nodes: np.ndarray,
    labels_by_index: Mapping[int, str],
) -> dict[int, Recommendation]:
    """Compute first-hit probabilities for one connected component."""

    node_list = [int(node) for node in nodes]
    labelled_nodes = [node for node in node_list if node in labels_by_index]
    if not labelled_nodes:
        return {
            node: Recommendation(
                status=STATUS_PENDING_LAB,
                material=None,
                probabilities={},
                reason=REASON_NO_REACHABLE_LAB,
            )
            for node in node_list
        }

    labelled_set = set(labelled_nodes)
    unlabelled = [node for node in node_list if node not in labelled_set]
    if not unlabelled:
        return {}

    materials = sorted({labels_by_index[node] for node in labelled_nodes})
    material_position = {material: index for index, material in enumerate(materials)}
    unlabelled_position = {node: index for index, node in enumerate(unlabelled)}

    size = len(unlabelled)
    matrix = np.eye(size, dtype=np.float64)
    rhs = np.zeros((size, len(materials)), dtype=np.float64)

    for node in unlabelled:
        row = unlabelled_position[node]
        neighbours = np.flatnonzero(adjacency[node])
        degree = int(neighbours.size)
        if degree == 0:
            # Cannot happen for a node in a component containing a label, but
            # guarding here avoids a division by zero if data is inconsistent.
            raise ComputationError("未化验布样没有邻居，无法进行随机游走")
        probability_step = 1.0 / float(degree)
        for neighbour in neighbours:
            neighbour = int(neighbour)
            if neighbour in labels_by_index:
                material = labels_by_index[neighbour]
                rhs[row, material_position[material]] += probability_step
            else:
                matrix[row, unlabelled_position[neighbour]] -= probability_step

    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ComputationError("随机游走首达概率方程组无唯一解") from exc

    if not np.all(np.isfinite(solution)):
        raise ComputationError("随机游走首达概率出现非有限值")

    residual = np.max(np.abs(matrix @ solution - rhs)) if rhs.size else 0.0
    if residual > RESIDUAL_TOLERANCE:
        raise ComputationError(f"首达概率方程残差过大：{residual:.3e}")

    if np.min(solution) < -RESIDUAL_TOLERANCE or np.max(solution) > 1.0 + RESIDUAL_TOLERANCE:
        raise ComputationError("首达概率超出 [0, 1] 范围")
    solution = np.clip(solution, 0.0, 1.0)

    absorption = solution.sum(axis=1)
    if np.any(np.abs(absorption - 1.0) > STATE_TOLERANCE):
        raise ComputationError("各材质首达概率之和不等于 1")

    recommendations: dict[int, Recommendation] = {}
    for node in unlabelled:
        row = unlabelled_position[node]
        probabilities = {
            material: float(solution[row, column])
            for column, material in enumerate(materials)
        }
        best_value = max(probabilities.values())
        # ``materials`` is sorted, so ties (within numerical noise) resolve to
        # the lexicographically smallest material deterministically.
        best_material = next(
            material
            for material in materials
            if probabilities[material] >= best_value - PROBABILITY_TOLERANCE
        )
        if best_value >= BIN_PROBABILITY - PROBABILITY_TOLERANCE:
            recommendation = Recommendation(
                status=STATUS_RECOMMEND_BIN,
                material=best_material,
                probabilities=probabilities,
                reason=None,
            )
        else:
            recommendation = Recommendation(
                status=STATUS_PENDING_LAB,
                material=None,
                probabilities=probabilities,
                reason=REASON_BELOW_THRESHOLD,
            )
        recommendations[node] = recommendation

    return recommendations


def compute_recommendations(
    adjacency: np.ndarray,
    components: Sequence[np.ndarray],
    labels_by_index: Mapping[int, str],
) -> dict[int, Recommendation]:
    """Compute recommendations for every unlabelled node in the graph."""

    recommendations: dict[int, Recommendation] = {}
    for component in components:
        recommendations.update(
            _solve_component(adjacency, component, labels_by_index)
        )
    return recommendations


def compute_component_recommendations(
    adjacency: np.ndarray,
    component: np.ndarray,
    labels_by_index: Mapping[int, str],
) -> dict[int, Recommendation]:
    return _solve_component(adjacency, component, labels_by_index)


def _probability(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def material_uncertainty(recommendation: Recommendation) -> float:
    """Return ``1 - max(probability)`` for an unlabelled sample.

    A sample whose component has no reachable lab point has no probability
    distribution at all; its uncertainty is defined as 1.0 so that it ranks
    as the least certain material assignment.
    """

    if not recommendation.probabilities:
        return 1.0
    highest = max(recommendation.probabilities.values())
    return min(1.0, max(0.0, 1.0 - float(highest)))


def _selection_key(uncertainty: float, sample_id: int) -> tuple[float, int]:
    # Rounding makes mathematically tied uncertainties compare equal even if
    # the solver produced last-bit differences; ties fall back to sample id.
    return (-round(float(uncertainty), 12), int(sample_id))


def compute_lab_queue(
    components: Sequence[np.ndarray],
    baseline: Mapping[int, Recommendation],
    sample_ids: Sequence[int],
    quota: int,
) -> dict:
    """Select unlabelled samples for proactive lab testing.

    Selection has two phases:

    1. Every connected component that has at least one unlabelled sample gets
       one sample (when the quota allows it).  Components are visited by their
       best candidate first, so a quota smaller than the component count still
       favours the least certain components.
    2. The remaining slots are filled globally by uncertainty descending and
       sample id ascending.

    The result is deterministic for a fixed graph, label set and quota.  The
    queue never contains duplicates and never exceeds either the quota or the
    number of unlabelled samples.
    """

    if quota < 1:
        raise ComputationError("送检名额必须为正整数")

    component_of: dict[int, int] = {}
    for component_index, component in enumerate(components):
        for node in component:
            component_of[int(node)] = component_index

    candidates: list[dict] = []
    for node, recommendation in baseline.items():
        if node not in component_of:
            raise ComputationError("未化验布样不在任何连通分量中")
        sample_id = int(sample_ids[node])
        uncertainty = material_uncertainty(recommendation)
        candidates.append(
            {
                "node": int(node),
                "sample_id": sample_id,
                "uncertainty": uncertainty,
                "component_index": component_of[node],
                "recommendation": recommendation,
            }
        )

    by_component: dict[int, list[dict]] = {}
    for candidate in candidates:
        by_component.setdefault(candidate["component_index"], []).append(candidate)

    for group in by_component.values():
        group.sort(
            key=lambda item: _selection_key(item["uncertainty"], item["sample_id"])
        )

    component_order = sorted(
        by_component,
        key=lambda component_index: _selection_key(
            by_component[component_index][0]["uncertainty"],
            by_component[component_index][0]["sample_id"],
        ),
    )

    selected: list[dict] = []
    selected_nodes: set[int] = set()

    for component_index in component_order:
        if len(selected) >= quota:
            break
        candidate = by_component[component_index][0]
        selected.append(candidate)
        selected_nodes.add(candidate["node"])

    remaining = sorted(
        (candidate for candidate in candidates if candidate["node"] not in selected_nodes),
        key=lambda item: _selection_key(item["uncertainty"], item["sample_id"]),
    )
    for candidate in remaining:
        if len(selected) >= quota:
            break
        selected.append(candidate)

    queue: list[dict] = []
    for position, candidate in enumerate(selected, start=1):
        recommendation = candidate["recommendation"]
        queue.append(
            {
                "position": position,
                "sample_id": candidate["sample_id"],
                "component_index": candidate["component_index"],
                "uncertainty": _probability(candidate["uncertainty"]),
                "probabilities": {
                    material: _probability(value)
                    for material, value in recommendation.probabilities.items()
                },
            }
        )

    return {
        "queue": queue,
        "candidate_count": len(candidates),
        "component_count": len(by_component),
        "covered_component_count": len(
            {candidate["component_index"] for candidate in selected}
        ),
    }


def _destination(recommendation: Recommendation) -> dict:
    """Serialise one recommendation as an operator-facing destination."""

    return recommendation.as_dict()


def compute_review(
    adjacency: np.ndarray,
    components: Sequence[np.ndarray],
    labels_by_index: Mapping[int, str],
    sample_ids: Sequence[int],
    baseline: Mapping[int, Recommendation],
    batch_id: str,
    version: int,
) -> dict:
    """Return the impact of temporarily withdrawing each lab conclusion.

    Only samples that were unlabelled and binnable before the withdrawal are
    counted.  A sample counts once when it moves to another bin or becomes
    pending; samples that merely change numeric probabilities do not count.
    """

    component_of: dict[int, int] = {}
    for component_index, component in enumerate(components):
        for node in component:
            component_of[int(node)] = component_index

    binnable = sorted(
        (node for node, recommendation in baseline.items()
         if recommendation.status == STATUS_RECOMMEND_BIN),
        key=lambda node: sample_ids[node],
    )
    binnable_by_component: dict[int, list[int]] = {}
    for node in binnable:
        binnable_by_component.setdefault(component_of[node], []).append(node)

    label_indices = sorted(labels_by_index, key=lambda node: sample_ids[node])
    candidates: list[dict] = []

    for removed in label_indices:
        component_index = component_of[removed]
        affected: list[dict] = []
        relevant_binnable = binnable_by_component.get(component_index, [])

        if relevant_binnable:
            new_labels = dict(labels_by_index)
            del new_labels[removed]
            after = compute_component_recommendations(
                adjacency, components[component_index], new_labels
            )
            for node in relevant_binnable:
                before_recommendation = baseline[node]
                after_recommendation = after[node]
                if (
                    after_recommendation.status == STATUS_RECOMMEND_BIN
                    and after_recommendation.material == before_recommendation.material
                ):
                    continue
                affected.append(
                    {
                        "sample_id": int(sample_ids[node]),
                        "before": _destination(before_recommendation),
                        "after": _destination(after_recommendation),
                    }
                )

        affected.sort(key=lambda item: item["sample_id"])
        candidates.append(
            {
                "sample_id": int(sample_ids[removed]),
                "material": labels_by_index[removed],
                "affected_count": len(affected),
                "affected_sample_ids": [item["sample_id"] for item in affected],
                "_affected": affected,
            }
        )

    winner: dict | None = None
    for candidate in candidates:
        if winner is None or candidate["affected_count"] > winner["affected_count"]:
            winner = candidate

    if winner is None or winner["affected_count"] == 0:
        priority_id = None
        priority_material = None
        affected_samples: list[dict] = []
        message = (
            "无需优先复核：暂撤任一化验结论都不会改变原先可入箱"
            "未化验布样的去向。"
        )
    else:
        priority_id = winner["sample_id"]
        priority_material = winner["material"]
        affected_samples = winner["_affected"]
        message = (
            f"建议优先复核已化验布样 {priority_id}（材质 {priority_material}）："
            f"暂撤该结论会使 {winner['affected_count']} 个原先可入箱的未化验布样"
            "改箱或转为待化验。"
        )

    public_candidates = [
        {key: value for key, value in candidate.items() if key != "_affected"}
        for candidate in candidates
    ]

    return {
        "batch_id": batch_id,
        "version": version,
        "priority_review_sample_id": priority_id,
        "priority_review_material": priority_material,
        "max_affected_count": winner["affected_count"] if winner else 0,
        "needs_priority_review": bool(winner and winner["affected_count"] > 0),
        "affected_samples": affected_samples,
        "candidates": public_candidates,
        "baseline_binnable_count": len(binnable),
        "message": message,
    }
