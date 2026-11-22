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
