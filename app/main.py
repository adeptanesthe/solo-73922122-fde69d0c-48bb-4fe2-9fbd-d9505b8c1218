"""FastAPI application for fabric sorting review."""

from __future__ import annotations

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import BatchCreateIn, LabelUpdateIn
from .store import StoreError, store

app = FastAPI(
    title="布料分拣复核 API",
    version="1.0.0",
    description=(
        "按归一化光谱向量的欧氏距离建图，随机游走计算首达化验点的材质概率，"
        "并评估逐条暂撤化验结论对分箱建议的影响。"
    ),
)


def _error(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details}},
    )


@app.exception_handler(StoreError)
async def _store_error_handler(_: Request, exc: StoreError) -> JSONResponse:
    return _error(exc.status_code, exc.code, exc.message, exc.details)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for error in exc.errors():
        item = dict(error)
        context = item.get("ctx")
        if isinstance(context, dict):
            item["ctx"] = {key: str(value) for key, value in context.items()}
        details.append(item)
    return _error(422, "validation_error", "请求数据校验失败", details)


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "http_error"
    return _error(exc.status_code, code, str(exc.detail))


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/batches", status_code=201, tags=["batches"])
def create_batch(payload: BatchCreateIn) -> dict:
    """Create a batch (≤100 samples) and compute initial recommendations."""

    return store.create(payload)


@app.get("/batches/{batch_id}", tags=["batches"])
def get_batch(batch_id: str) -> dict:
    """Return labels and recommendations with the current batch version."""

    return store.snapshot(batch_id)


@app.put("/batches/{batch_id}/samples/{sample_id}/label", tags=["labels"])
def put_label(batch_id: str, sample_id: int, payload: LabelUpdateIn) -> dict:
    """Add or replace a lab conclusion; labels and recommendations update together."""

    return store.put_label(batch_id, sample_id, payload.material, payload.version)


@app.delete("/batches/{batch_id}/samples/{sample_id}/label", tags=["labels"])
def delete_label(
    batch_id: str,
    sample_id: int,
    version: int = Query(ge=1, description="提交时看到的批次版号"),
) -> dict:
    """Withdraw a lab conclusion; labels and recommendations update together."""

    return store.delete_label(batch_id, sample_id, version)


@app.get("/batches/{batch_id}/review", tags=["review"])
def review_batch(batch_id: str) -> dict:
    """Identify the single lab conclusion whose withdrawal affects the most bins.

    Only originally binnable, unlabelled samples that change bin or become
    pending are counted.  The response carries the batch version it was
    computed against.
    """

    return store.record_review(batch_id)
