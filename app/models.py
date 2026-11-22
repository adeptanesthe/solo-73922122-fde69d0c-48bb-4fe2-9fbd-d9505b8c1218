"""Input/output models for the fabric sorting API."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

MAX_SAMPLES = 100
MAX_LAB_QUEUE = 10
SAMPLE_ID_DESCRIPTION = "布样整数编号，批次内唯一"


class SampleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(description=SAMPLE_ID_DESCRIPTION)
    vector: list[float] = Field(
        min_length=1,
        description="已归一化的光谱向量；距离按提交值直接计算",
    )

    @field_validator("vector")
    @classmethod
    def _finite_vector(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(component) for component in value):
            raise ValueError("光谱向量分量必须为有限数值")
        norm = math.sqrt(sum(component * component for component in value))
        if norm <= 0.0:
            raise ValueError("光谱向量不能是零向量")
        return value


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: int
    material: str = Field(min_length=1, max_length=128)

    @field_validator("material")
    @classmethod
    def _clean_material(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("材质名称不能为空白")
        return cleaned


class BatchCreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="可选的业务批次号；不填由服务端生成",
    )
    threshold: float = Field(
        ge=0.0,
        description="欧氏距离阈值；距离不超过阈值的布样相连",
    )
    samples: Annotated[list[SampleIn], Field(min_length=1, max_length=MAX_SAMPLES)]
    labels: list[LabelIn] = Field(
        default_factory=list,
        description="建批时已有的材质化验结论",
    )

    @field_validator("threshold")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("距离阈值必须为有限数值")
        return value

    @model_validator(mode="after")
    def _unique_samples_and_consistent_labels(self) -> "BatchCreateIn":
        sample_ids = [sample.id for sample in self.samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("布样编号必须唯一")
        known = set(sample_ids)
        seen_labels: set[int] = set()
        for label in self.labels:
            if label.sample_id not in known:
                raise ValueError(f"化验结论引用了不存在的布样编号 {label.sample_id}")
            if label.sample_id in seen_labels:
                raise ValueError(f"布样 {label.sample_id} 有多条化验结论")
            seen_labels.add(label.sample_id)
        dimension = len(self.samples[0].vector)
        if any(len(sample.vector) != dimension for sample in self.samples):
            raise ValueError("所有光谱向量的维度必须一致")
        return self


class LabelUpdateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1, description="提交时看到的批次版号")

    @field_validator("material")
    @classmethod
    def _clean_material(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("材质名称不能为空白")
        return cleaned


class LabQueueIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1, description="提交时看到的批次版号")
    quota: int = Field(
        ge=1,
        le=MAX_LAB_QUEUE,
        validation_alias=AliasChoices("quota", "count"),
        description=f"本次主动送检名额，1–{MAX_LAB_QUEUE}",
    )


class ErrorBody(BaseModel):
    code: str
    message: str
    details: object | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
