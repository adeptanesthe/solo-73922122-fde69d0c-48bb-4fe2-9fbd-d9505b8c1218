"""Versioned in-memory batch store with optimistic concurrency control."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import engine
from .models import BatchCreateIn


class StoreError(Exception):
    """Base error carrying an API error code and HTTP status."""

    status_code = 400
    code = "invalid_request"

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class BatchNotFound(StoreError):
    status_code = 404
    code = "batch_not_found"


class BatchExists(StoreError):
    status_code = 409
    code = "batch_exists"


class SampleNotFound(StoreError):
    status_code = 404
    code = "sample_not_found"


class VersionConflict(StoreError):
    status_code = 409
    code = "version_conflict"

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message, details)


class LabelAbsent(StoreError):
    status_code = 409
    code = "label_absent"


class ComputationFailed(StoreError):
    status_code = 500
    code = "computation_failed"


@dataclass
class Batch:
    batch_id: str
    threshold: float
    sample_ids: list[int]
    vectors: np.ndarray
    labels: dict[int, str]
    adjacency: np.ndarray
    components: list[np.ndarray]
    version: int = 1
    baseline: dict[int, engine.Recommendation] = field(default_factory=dict)
    last_review: dict | None = None
    review_fingerprint: tuple | None = None

    def sample_index(self) -> dict[int, int]:
        return {sample_id: index for index, sample_id in enumerate(self.sample_ids)}

    def labels_by_index(self) -> dict[int, str]:
        index_of = self.sample_index()
        return {
            index_of[sample_id]: material
            for sample_id, material in self.labels.items()
        }

    def snapshot(self) -> dict:
        index_of = self.sample_index()
        recommendations = [
            {
                "sample_id": sample_id,
                **self.baseline[index_of[sample_id]].as_dict(),
            }
            for sample_id in sorted(self.sample_ids)
            if index_of[sample_id] in self.baseline
        ]
        binnable = sum(
            1
            for item in recommendations
            if item["status"] == engine.STATUS_RECOMMEND_BIN
        )
        return {
            "batch_id": self.batch_id,
            "version": self.version,
            "threshold": self.threshold,
            "sample_count": len(self.sample_ids),
            "labelled_sample_ids": sorted(self.labels),
            "labels": [
                {"sample_id": sample_id, "material": material}
                for sample_id, material in sorted(self.labels.items())
            ],
            "unlabelled_count": len(self.sample_ids) - len(self.labels),
            "recommend_bin_count": binnable,
            "pending_lab_count": len(recommendations) - binnable,
            "recommendations": recommendations,
        }


class BatchStore:
    def __init__(self) -> None:
        self._batches: dict[str, Batch] = {}
        self._lock = threading.RLock()

    def _recompute(self, batch: Batch) -> None:
        try:
            batch.baseline = engine.compute_recommendations(
                batch.adjacency, batch.components, batch.labels_by_index()
            )
        except engine.ComputationError as exc:
            raise ComputationFailed(str(exc)) from exc

    def create(self, payload: BatchCreateIn) -> dict:
        batch_id = payload.batch_id or f"batch-{uuid.uuid4().hex[:12]}"
        vectors = np.asarray([sample.vector for sample in payload.samples], dtype=np.float64)
        sample_ids = [sample.id for sample in payload.samples]
        try:
            adjacency = engine.build_adjacency(vectors, payload.threshold)
        except engine.ComputationError as exc:
            raise ComputationFailed(str(exc)) from exc
        components = engine.find_components(adjacency)

        batch = Batch(
            batch_id=batch_id,
            threshold=float(payload.threshold),
            sample_ids=sample_ids,
            vectors=vectors,
            labels={label.sample_id: label.material for label in payload.labels},
            adjacency=adjacency,
            components=components,
        )
        self._recompute(batch)

        with self._lock:
            if batch_id in self._batches:
                raise BatchExists(
                    f"批次 {batch_id} 已存在", {"batch_id": batch_id}
                )
            self._batches[batch_id] = batch
            return batch.snapshot()

    def get(self, batch_id: str) -> Batch:
        with self._lock:
            batch = self._batches.get(batch_id)
        if batch is None:
            raise BatchNotFound(f"批次 {batch_id} 不存在")
        return batch

    def _require_sample(self, batch: Batch, sample_id: int) -> None:
        if sample_id not in batch.sample_ids:
            raise SampleNotFound(f"布样 {sample_id} 不在批次 {batch.batch_id} 中")

    def _check_version(self, batch: Batch, version: int) -> None:
        if version != batch.version:
            raise VersionConflict(
                f"批次版号不匹配：提交 {version}，当前 {batch.version}，请刷新后重试",
                {"submitted_version": version, "current_version": batch.version},
            )

    def snapshot(self, batch_id: str) -> dict:
        with self._lock:
            return self.get(batch_id).snapshot()

    def put_label(self, batch_id: str, sample_id: int, material: str, version: int) -> dict:
        with self._lock:
            batch = self.get(batch_id)
            self._require_sample(batch, sample_id)
            self._check_version(batch, version)

            previous_labels = dict(batch.labels)
            previous_baseline = dict(batch.baseline)
            previous_version = batch.version
            batch.labels[sample_id] = material
            try:
                self._recompute(batch)
            except StoreError:
                batch.labels = previous_labels
                batch.baseline = previous_baseline
                batch.version = previous_version
                raise
            batch.version += 1
            return batch.snapshot()

    def delete_label(self, batch_id: str, sample_id: int, version: int) -> dict:
        with self._lock:
            batch = self.get(batch_id)
            self._require_sample(batch, sample_id)
            self._check_version(batch, version)
            if sample_id not in batch.labels:
                raise LabelAbsent(f"布样 {sample_id} 当前没有化验结论")

            previous_labels = dict(batch.labels)
            previous_baseline = dict(batch.baseline)
            previous_version = batch.version
            del batch.labels[sample_id]
            try:
                self._recompute(batch)
            except StoreError:
                batch.labels = previous_labels
                batch.baseline = previous_baseline
                batch.version = previous_version
                raise
            batch.version += 1
            return batch.snapshot()

    def review(self, batch_id: str) -> dict:
        with self._lock:
            batch = self.get(batch_id)
            try:
                return engine.compute_review(
                    adjacency=batch.adjacency,
                    components=batch.components,
                    labels_by_index=batch.labels_by_index(),
                    sample_ids=batch.sample_ids,
                    baseline=batch.baseline,
                    batch_id=batch.batch_id,
                    version=batch.version,
                )
            except engine.ComputationError as exc:
                raise ComputationFailed(str(exc)) from exc

    def record_review(self, batch_id: str) -> dict:
        """Run the review and remember it for this batch version and state.

        The fingerprint covers the label set and the version, so a cached
        result can never be served after a label change.
        """

        with self._lock:
            batch = self.get(batch_id)
            fingerprint = (batch.version, tuple(sorted(batch.labels.items())))
            if batch.last_review is not None and batch.review_fingerprint == fingerprint:
                return batch.last_review
            result = self.review(batch_id)
            batch.last_review = result
            batch.review_fingerprint = fingerprint
            return result

    def clear(self) -> None:
        with self._lock:
            self._batches.clear()


store = BatchStore()
