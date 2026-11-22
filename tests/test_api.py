"""API-level tests: versioning, review impact and conflict handling."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import store


@pytest.fixture(autouse=True)
def _clean_store():
    store.clear()
    yield
    store.clear()


@pytest.fixture()
def client():
    return TestClient(app)


def _unit(angle_degrees: float) -> list[float]:
    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians)]


# Samples placed on a circle.  Distances: 10° between neighbours, so with a
# threshold of 0.2 (chord length of ~11.5°) only adjacent samples connect.
SAMPLES = [
    {"id": 1, "vector": _unit(0)},
    {"id": 2, "vector": _unit(10)},
    {"id": 3, "vector": _unit(20)},
    {"id": 4, "vector": _unit(30)},
    {"id": 5, "vector": _unit(40)},
    {"id": 6, "vector": _unit(50)},
    {"id": 7, "vector": _unit(60)},
]


def _create(client: TestClient, labels=None, threshold: float = 0.2) -> dict:
    payload = {
        "batch_id": "B-1",
        "threshold": threshold,
        "samples": SAMPLES,
        "labels": labels or [],
    }
    response = client.post("/batches", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_batch_reports_version_and_recommendations(client):
    body = _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    assert body["version"] == 1
    assert body["labelled_sample_ids"] == [1]
    assert body["unlabelled_count"] == 6
    statuses = {item["sample_id"]: item["status"] for item in body["recommendations"]}
    # Labelled samples do not receive recommendations.
    assert set(statuses) == {2, 3, 4, 5, 6, 7}
    assert all(status in {"recommend_bin", "pending_lab"} for status in statuses.values())


def test_review_identifies_most_influential_label(client):
    # One path 1(cotton) -- 2 -- 3 -- 4(polyester) -- 5 -- 6 -- 7.
    body = _create(
        client,
        labels=[
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 4, "material": "polyester"},
        ],
    )
    baseline = {item["sample_id"]: item for item in body["recommendations"]}
    binnable = {
        sample_id
        for sample_id, item in baseline.items()
        if item["status"] == "recommend_bin"
    }
    assert binnable, "fixture should contain at least one binnable sample"

    review = client.get("/batches/B-1/review").json()
    assert review["version"] == 1
    assert review["batch_id"] == "B-1"
    assert review["baseline_binnable_count"] == len(binnable)

    counts = {item["sample_id"]: item["affected_count"] for item in review["candidates"]}
    assert set(counts) == {1, 4}
    expected_winner = min(counts, key=lambda sid: (-counts[sid], sid))
    if counts[expected_winner] == 0:
        assert review["needs_priority_review"] is False
        assert review["priority_review_sample_id"] is None
        assert review["affected_samples"] == []
    else:
        assert review["needs_priority_review"] is True
        assert review["priority_review_sample_id"] == expected_winner
        assert review["max_affected_count"] == counts[expected_winner]

    for candidate in review["candidates"]:
        for affected in candidate.get("affected_sample_ids", []):
            assert affected in binnable


def test_review_counts_only_original_binnable_and_bin_change_or_pending(client):
    body = _create(
        client,
        labels=[
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 4, "material": "polyester"},
        ],
    )
    baseline = {item["sample_id"]: item for item in body["recommendations"]}
    review = client.get("/batches/B-1/review").json()

    for candidate in review["candidates"]:
        # candidates only list labels that were on the batch, never sample 0.
        assert candidate["material"] in {"cotton", "polyester"}

    winner_id = review["priority_review_sample_id"]
    if winner_id is not None:
        winner = next(c for c in review["candidates"] if c["sample_id"] == winner_id)
        # The winner must be the smallest id among equally impactful labels.
        tied = [
            c["sample_id"]
            for c in review["candidates"]
            if c["affected_count"] == winner["affected_count"]
        ]
        assert winner_id == min(tied)
        for affected in review["affected_samples"]:
            before = affected["before"]
            after = affected["after"]
            assert before["status"] == baseline[affected["sample_id"]]["status"]
            assert before["material"] == baseline[affected["sample_id"]]["material"]
            assert not (
                after["status"] == "recommend_bin"
                and after["material"] == before["material"]
            )


def test_review_reports_no_priority_when_nothing_changes(client):
    # Two cotton labels both reach the only unlabelled sample.  Removing
    # either one still leaves a cotton lab point, so the bin destination is
    # unchanged and no priority review is needed.
    payload = {
        "batch_id": "B-2",
        "threshold": 0.05,
        "samples": [
            {"id": 1, "vector": _unit(0)},
            {"id": 2, "vector": _unit(4)},
            {"id": 3, "vector": _unit(2)},
        ],
        "labels": [
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 2, "material": "cotton"},
        ],
    }
    created = client.post("/batches", json=payload)
    assert created.status_code == 201, created.text
    review = client.get("/batches/B-2/review").json()
    assert review["version"] == 1
    assert review["baseline_binnable_count"] == 1
    assert review["needs_priority_review"] is False
    assert review["priority_review_sample_id"] is None
    assert review["max_affected_count"] == 0
    assert review["affected_samples"] == []
    assert "无需优先复核" in review["message"]


def test_review_reports_bin_flip_when_a_label_is_withdrawn(client):
    # Samples at 0°, 15°, 30°, 45°, 60° on a circle.  With a threshold of 0.6
    # (chord length 0.517 for 30°), labels at 0° (cotton) and 30° (wool):
    # removing the wool label flips the 45° and 60° samples from wool to
    # cotton (their wool probabilities were 0.846 and 0.923).
    samples = [
        {"id": 1, "vector": _unit(0)},
        {"id": 2, "vector": _unit(15)},
        {"id": 3, "vector": _unit(30)},
        {"id": 4, "vector": _unit(45)},
        {"id": 5, "vector": _unit(60)},
    ]
    payload = {
        "batch_id": "B-3",
        "threshold": 0.6,
        "samples": samples,
        "labels": [
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 3, "material": "wool"},
        ],
    }
    created = client.post("/batches", json=payload)
    assert created.status_code == 201, created.text
    baseline = {item["sample_id"]: item for item in created.json()["recommendations"]}
    assert baseline[4]["status"] == "recommend_bin"
    assert baseline[4]["material"] == "wool"
    assert baseline[4]["probabilities"]["wool"] == pytest.approx(0.846153846154, abs=1e-9)

    review = client.get("/batches/B-3/review").json()
    assert review["version"] == 1
    assert review["needs_priority_review"] is True
    assert review["priority_review_sample_id"] == 3
    assert review["priority_review_material"] == "wool"
    assert review["max_affected_count"] == 2
    assert [item["sample_id"] for item in review["affected_samples"]] == [4, 5]
    for affected in review["affected_samples"]:
        assert affected["before"]["status"] == "recommend_bin"
        assert affected["before"]["material"] == "wool"
        assert affected["after"]["status"] == "recommend_bin"
        assert affected["after"]["material"] == "cotton"


def test_review_breaks_ties_by_smallest_sample_id(client):
    # Two disconnected clusters, each with one label and two unlabelled
    # samples.  Removing either label strands exactly two binnable samples,
    # so the impact is tied and the smaller sample id must be reported.
    samples = [
        {"id": 10, "vector": _unit(0)},
        {"id": 11, "vector": _unit(1)},
        {"id": 12, "vector": _unit(2)},
        {"id": 20, "vector": _unit(180)},
        {"id": 21, "vector": _unit(179)},
        {"id": 22, "vector": _unit(178)},
    ]
    payload = {
        "batch_id": "B-4",
        "threshold": 0.3,
        "samples": samples,
        "labels": [
            {"sample_id": 10, "material": "cotton"},
            {"sample_id": 20, "material": "wool"},
        ],
    }
    created = client.post("/batches", json=payload)
    assert created.status_code == 201, created.text
    assert created.json()["recommend_bin_count"] == 4

    review = client.get("/batches/B-4/review").json()
    counts = {c["sample_id"]: c["affected_count"] for c in review["candidates"]}
    assert counts == {10: 2, 20: 2}
    assert review["needs_priority_review"] is True
    assert review["max_affected_count"] == 2
    assert review["priority_review_sample_id"] == 10
    assert review["priority_review_material"] == "cotton"
    assert [item["sample_id"] for item in review["affected_samples"]] == [11, 12]
    for affected in review["affected_samples"]:
        assert affected["after"]["status"] == "pending_lab"
        assert affected["after"]["reason"] == "no_reachable_lab_point"


def test_review_reflects_label_change_and_new_version(client):
    body = _create(
        client,
        labels=[
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 4, "material": "polyester"},
        ],
    )
    first = client.get("/batches/B-1/review").json()
    assert first["version"] == body["version"]

    # Withdraw the polyester label; the review must not serve the cached
    # result for the old version.
    updated = client.delete("/batches/B-1/samples/4/label", params={"version": 1})
    assert updated.status_code == 200
    assert updated.json()["version"] == 2

    second = client.get("/batches/B-1/review").json()
    assert second["version"] == 2
    assert [c["sample_id"] for c in second["candidates"]] == [1]
    baseline = {item["sample_id"]: item for item in updated.json()["recommendations"]}
    assert second["baseline_binnable_count"] == sum(
        1 for item in baseline.values() if item["status"] == "recommend_bin"
    )
    # Only the current label can be the priority review target.
    assert second["priority_review_sample_id"] in (None, 1)


def test_label_put_and_delete_require_current_version(client):
    body = _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    assert body["version"] == 1

    stale = client.put(
        "/batches/B-1/samples/4/label",
        json={"material": "polyester", "version": 99},
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "version_conflict"

    updated = client.put(
        "/batches/B-1/samples/4/label",
        json={"material": "polyester", "version": 1},
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == 2
    assert {"sample_id": 4, "material": "polyester"} in updated.json()["labels"]

    stale_delete = client.delete("/batches/B-1/samples/1/label", params={"version": 1})
    assert stale_delete.status_code == 409

    removed = client.delete("/batches/B-1/samples/1/label", params={"version": 2})
    assert removed.status_code == 200
    assert removed.json()["version"] == 3
    assert 1 not in removed.json()["labelled_sample_ids"]


def test_label_and_recommendations_update_together(client):
    body = _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    before = {item["sample_id"]: item for item in body["recommendations"]}
    assert before[7]["material"] == "cotton"

    updated = client.put(
        "/batches/B-1/samples/7/label",
        json={"material": "wool", "version": 1},
    ).json()
    after = {item["sample_id"]: item for item in updated["recommendations"]}

    assert 7 not in after  # a labelled sample no longer receives a suggestion
    assert 7 in updated["labelled_sample_ids"]
    for sample_id, item in after.items():
        assert item["status"] in {"recommend_bin", "pending_lab"}
        if item["status"] == "recommend_bin":
            assert item["material"] is not None
            assert item["probabilities"][item["material"]] >= 0.8 - 1e-9
        else:
            assert item["material"] is None
            assert item["reason"] in {"below_threshold", "no_reachable_lab_point"}


def test_delete_missing_label_conflicts(client):
    _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    response = client.delete("/batches/B-1/samples/2/label", params={"version": 1})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "label_absent"


def test_failed_recomputation_keeps_previous_version(client, monkeypatch):
    body = _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    baseline_before = body["recommendations"]
    assert body["version"] == 1

    def _boom(*_args, **_kwargs):
        from app import engine

        raise engine.ComputationError("simulated solver failure")

    monkeypatch.setattr("app.store.engine.compute_recommendations", _boom)
    response = client.put(
        "/batches/B-1/samples/4/label",
        json={"material": "polyester", "version": 1},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "computation_failed"
    monkeypatch.undo()

    # The failed write must not bump the version, add the label, or alter the
    # recommendations that the operator still sees.
    snapshot = store.get("B-1")
    assert snapshot.version == 1
    assert 4 not in snapshot.labels
    assert client.get("/batches/B-1").json()["recommendations"] == baseline_before


def test_failed_recomputation_on_delete_restores_state(client, monkeypatch):
    _create(client, labels=[{"sample_id": 1, "material": "cotton"}])

    def _boom(*_args, **_kwargs):
        from app import engine

        raise engine.ComputationError("simulated solver failure")

    monkeypatch.setattr("app.store.engine.compute_recommendations", _boom)
    response = client.delete("/batches/B-1/samples/1/label", params={"version": 1})
    assert response.status_code == 500
    monkeypatch.undo()

    snapshot = store.get("B-1")
    assert snapshot.version == 1
    assert snapshot.labels == {1: "cotton"}


def test_put_replaces_existing_label_material(client):
    _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    response = client.put(
        "/batches/B-1/samples/1/label",
        json={"material": "linen", "version": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 2
    assert {"sample_id": 1, "material": "linen"} in body["labels"]
    assert not any(item["material"] == "cotton" for item in body["labels"])


def test_hundred_samples_are_accepted(client):
    samples = [{"id": index, "vector": _unit(index * 3.6)} for index in range(1, 101)]
    payload = {
        "batch_id": "B-100",
        "threshold": 0.2,
        "samples": samples,
        "labels": [
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 50, "material": "wool"},
            {"sample_id": 100, "material": "silk"},
        ],
    }
    created = client.post("/batches", json=payload)
    assert created.status_code == 201, created.text
    review = client.get("/batches/B-100/review").json()
    assert review["version"] == 1
    assert len(review["candidates"]) == 3


def test_validation_rejects_duplicate_ids_and_unknown_labels(client):
    duplicate = {
        "threshold": 0.2,
        "samples": [{"id": 1, "vector": [1, 0]}, {"id": 1, "vector": [0, 1]}],
    }
    assert client.post("/batches", json=duplicate).status_code == 422

    unknown_label = {
        "threshold": 0.2,
        "samples": [{"id": 1, "vector": [1, 0]}],
        "labels": [{"sample_id": 2, "material": "cotton"}],
    }
    assert client.post("/batches", json=unknown_label).status_code == 422

    too_many = {
        "threshold": 0.2,
        "samples": [{"id": i, "vector": [1, 0]} for i in range(101)],
    }
    assert client.post("/batches", json=too_many).status_code == 422


def test_unknown_batch_returns_404(client):
    assert client.get("/batches/missing").status_code == 404
    assert client.get("/batches/missing/review").status_code == 404


def test_duplicate_batch_id_conflicts(client):
    _create(client)
    duplicate = client.post(
        "/batches",
        json={"batch_id": "B-1", "threshold": 0.2, "samples": SAMPLES},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "batch_exists"


def test_label_endpoints_reject_unknown_sample(client):
    _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    put = client.put(
        "/batches/B-1/samples/999/label",
        json={"material": "cotton", "version": 1},
    )
    assert put.status_code == 404
    assert put.json()["error"]["code"] == "sample_not_found"
    delete = client.delete("/batches/B-1/samples/999/label", params={"version": 1})
    assert delete.status_code == 404


def test_version_must_be_positive(client):
    _create(client, labels=[{"sample_id": 1, "material": "cotton"}])
    response = client.delete("/batches/B-1/samples/1/label", params={"version": 0})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Proactive lab queue
# ---------------------------------------------------------------------------

# Three components with a threshold of 0.04:
#   [1,2,3,4] chain with cotton (1) and wool (4) labels,
#   [5,6]     chain without any lab point,
#   [7,8]     chain with a cotton label on 7.
# Uncertainties: 2 and 3 both 1/3, 5 and 6 both 1.0, 8 is 0.0.
QUEUE_SAMPLES = [
    {"id": 1, "vector": _unit(0)},
    {"id": 2, "vector": _unit(2)},
    {"id": 3, "vector": _unit(4)},
    {"id": 4, "vector": _unit(6)},
    {"id": 5, "vector": _unit(90)},
    {"id": 6, "vector": _unit(92)},
    {"id": 7, "vector": _unit(180)},
    {"id": 8, "vector": _unit(182)},
]


def _create_queue_batch(client: TestClient, batch_id: str = "Q-1") -> dict:
    payload = {
        "batch_id": batch_id,
        "threshold": 0.04,
        "samples": QUEUE_SAMPLES,
        "labels": [
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 4, "material": "wool"},
            {"sample_id": 7, "material": "cotton"},
        ],
    }
    response = client.post("/batches", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _queue(client: TestClient, batch_id: str, version: int, quota: int) -> dict:
    return client.post(
        f"/batches/{batch_id}/lab-queue",
        json={"version": version, "quota": quota},
    )


def test_lab_queue_returns_version_bound_selection(client):
    body = _create_queue_batch(client)
    response = _queue(client, "Q-1", body["version"], 3)
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["batch_id"] == "Q-1"
    assert result["version"] == 1
    assert result["quota"] == 3
    assert result["selected_count"] == 3
    assert result["unlabelled_count"] == 5
    assert result["component_count"] == 3
    assert result["covered_component_count"] == 3
    # One slot per component first: 5 (label-free), 2 (lowest id among the
    # 1/3 ties), then 8 (certain but the only unlabelled sample in its
    # component).  Node 6 stays out because node 5 already covers it.
    assert result["sample_ids"] == [5, 2, 8]
    assert [item["position"] for item in result["samples"]] == [1, 2, 3]
    assert result["samples"][0]["uncertainty"] == 1.0
    assert result["samples"][0]["probabilities"] == {}
    assert result["samples"][1]["uncertainty"] == pytest.approx(1 / 3)
    assert result["samples"][1]["probabilities"] == {
        "cotton": pytest.approx(0.666666666667),
        "wool": pytest.approx(0.333333333333),
    }
    assert result["samples"][2]["uncertainty"] == 0.0


def test_lab_queue_fills_remaining_slots_by_uncertainty_then_id(client):
    _create_queue_batch(client)
    result = _queue(client, "Q-1", 1, 5).json()
    # After one sample per component, the spare slots go to node 6 (1.0)
    # then node 3 (1/3, tied with node 2 which is already queued).
    assert result["sample_ids"] == [5, 2, 8, 6, 3]
    assert result["selected_count"] == 5


def test_lab_queue_same_version_and_quota_is_idempotent(client):
    _create_queue_batch(client)
    first = _queue(client, "Q-1", 1, 4)
    second = _queue(client, "Q-1", 1, 4)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    # Same version but a different quota is a different selection request.
    other = _queue(client, "Q-1", 1, 2).json()
    assert other["sample_ids"] == [5, 2]
    assert other["quota"] == 2


def test_lab_queue_rejects_stale_version(client):
    _create_queue_batch(client)
    updated = client.put(
        "/batches/Q-1/samples/2/label",
        json={"material": "cotton", "version": 1},
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == 2

    stale = _queue(client, "Q-1", 1, 2)
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "version_conflict"
    assert stale.json()["error"]["details"] == {
        "submitted_version": 1,
        "current_version": 2,
    }

    fresh = _queue(client, "Q-1", 2, 2)
    assert fresh.status_code == 200
    assert fresh.json()["version"] == 2
    # Sample 2 is now labelled and must not be queued any more.
    assert 2 not in fresh.json()["sample_ids"]


def test_lab_queue_cache_is_invalidated_by_label_change(client):
    _create_queue_batch(client)
    before = _queue(client, "Q-1", 1, 3).json()
    assert before["sample_ids"] == [5, 2, 8]

    changed = client.put(
        "/batches/Q-1/samples/2/label",
        json={"material": "cotton", "version": 1},
    )
    assert changed.status_code == 200
    assert changed.json()["version"] == 2

    after = _queue(client, "Q-1", 2, 3).json()
    assert after["sample_ids"] != before["sample_ids"]
    assert 2 not in after["sample_ids"]
    assert 3 in after["sample_ids"]


def test_lab_queue_cache_is_invalidated_by_label_withdrawal(client):
    _create_queue_batch(client)
    before = _queue(client, "Q-1", 1, 2).json()
    assert before["sample_ids"] == [5, 2]

    removed = client.delete("/batches/Q-1/samples/7/label", params={"version": 1})
    assert removed.status_code == 200
    assert removed.json()["version"] == 2

    after = _queue(client, "Q-1", 2, 2).json()
    # The whole [7, 8] component became label-free, turning both samples into
    # uncertainty-1.0 candidates; sample 7 is unlabelled now as well.  The two
    # components tie at 1.0, so the smaller component representative wins.
    assert after["sample_ids"] == [5, 7]
    assert 7 not in before["sample_ids"]


def test_lab_queue_records_review_and_queue_together_in_snapshot(client):
    _create_queue_batch(client)
    first_queue = _queue(client, "Q-1", 1, 2).json()
    first_review = client.get("/batches/Q-1/review").json()

    updated = client.put(
        "/batches/Q-1/samples/8/label",
        json={"material": "wool", "version": 1},
    ).json()
    assert updated["version"] == 2

    # Neither cached artefact may be served after the label change.
    second_review = client.get("/batches/Q-1/review").json()
    assert second_review["version"] == 2
    second_queue = _queue(client, "Q-1", 2, 2).json()
    assert second_queue["version"] == 2
    # Sample 8 is labelled now, so its component's representative changes to
    # the next candidate (none left -> the component stops contributing).
    assert first_queue["sample_ids"] == [5, 2]
    assert second_queue["sample_ids"] == [5, 2]
    assert first_review["version"] == 1


def test_lab_queue_validation(client):
    _create_queue_batch(client)
    # Quota bounds are inclusive 1..10.
    assert _queue(client, "Q-1", 1, 0).status_code == 422
    assert _queue(client, "Q-1", 1, 11).status_code == 422
    assert _queue(client, "Q-1", 0, 1).status_code == 422
    # Missing fields are rejected with the shared validation error envelope.
    missing = client.post("/batches/Q-1/lab-queue", json={"version": 1})
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "validation_error"
    # Unknown batch is a 404.
    assert _queue(client, "missing", 1, 1).status_code == 404
    assert _queue(client, "missing", 1, 1).json()["error"]["code"] == "batch_not_found"


def test_lab_queue_failed_computation_keeps_state(client, monkeypatch):
    body = _create_queue_batch(client)

    def _boom(*_args, **_kwargs):
        from app import engine

        raise engine.ComputationError("simulated queue failure")

    monkeypatch.setattr("app.store.engine.compute_lab_queue", _boom)
    response = _queue(client, "Q-1", 1, 2)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "computation_failed"
    # A failed selection must not cache a result nor bump the version.
    assert store.get("Q-1").version == body["version"]
    assert store.get("Q-1").lab_queues == {}
    monkeypatch.undo()

    retried = _queue(client, "Q-1", 1, 2)
    assert retried.status_code == 200
    assert retried.json()["selected_count"] == 2


def test_lab_queue_failed_recomputation_does_not_strand_a_stale_queue(client, monkeypatch):
    _create_queue_batch(client)
    first = _queue(client, "Q-1", 1, 2).json()

    def _boom(*_args, **_kwargs):
        from app import engine

        raise engine.ComputationError("simulated solver failure")

    monkeypatch.setattr("app.store.engine.compute_recommendations", _boom)
    failed = client.put(
        "/batches/Q-1/samples/2/label",
        json={"material": "wool", "version": 1},
    )
    assert failed.status_code == 500
    monkeypatch.undo()

    # Because the label write rolled back, the old (version, quota) queue is
    # still the correct queue for the unchanged state.
    assert store.get("Q-1").version == 1
    assert _queue(client, "Q-1", 1, 2).json() == first


def test_lab_queue_concurrent_retries_are_identical_and_bounded(client):
    import threading

    _create_queue_batch(client)
    responses: list[dict] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            response = _queue(client, "Q-1", 1, 4)
            responses.append(response.json())
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert len(responses) == 8
    first = responses[0]
    assert first["version"] == 1
    assert len(first["sample_ids"]) == 4 == len(set(first["sample_ids"]))
    assert all(response == first for response in responses)


def test_lab_queue_concurrent_requests_with_mixed_versions_never_exceed_quota(client):
    import threading

    _create_queue_batch(client)
    results: list[tuple[int, dict]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(6)

    def worker(index: int) -> None:
        barrier.wait(timeout=5)
        response = _queue(client, "Q-1", 1, 3)
        with lock:
            results.append((response.status_code, response.json()))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 6
    for status, body in results:
        assert status == 200
        assert body["quota"] == 3
        assert body["selected_count"] == 3
        assert len(set(body["sample_ids"])) == 3
    assert len({tuple(body["sample_ids"]) for _, body in results}) == 1


def test_lab_queue_truncates_when_quota_exceeds_candidates(client):
    _create_queue_batch(client)
    result = _queue(client, "Q-1", 1, 10).json()
    # Five unlabelled samples exist; the endpoint never invents extra slots.
    assert result["selected_count"] == 5
    assert result["quota"] == 10
    assert result["sample_ids"] == [5, 2, 8, 6, 3]
    assert "只能选出 5 份" in result["message"]


def test_lab_queue_with_no_unlabelled_samples_is_empty(client):
    payload = {
        "batch_id": "Q-FULL",
        "threshold": 0.2,
        "samples": [
            {"id": 1, "vector": _unit(0)},
            {"id": 2, "vector": _unit(5)},
        ],
        "labels": [
            {"sample_id": 1, "material": "cotton"},
            {"sample_id": 2, "material": "wool"},
        ],
    }
    assert client.post("/batches", json=payload).status_code == 201
    response = _queue(client, "Q-FULL", 1, 3)
    assert response.status_code == 200
    body = response.json()
    assert body["selected_count"] == 0
    assert body["sample_ids"] == []
    assert body["samples"] == []
    assert body["covered_component_count"] == 0
    # The empty queue is still cached and idempotent.
    assert _queue(client, "Q-FULL", 1, 3).json() == body
