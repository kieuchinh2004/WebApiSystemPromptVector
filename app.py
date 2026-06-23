"""
app.py
FastAPI application — định nghĩa các endpoint và schema cho WebAPI_Vector.
"""
import asyncio
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import classifier
import config as cfg
import llama_manager
from vector_schema import vector_detail

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prompt Classifier API (Vector + APC v4)",
    description=(
        "Web API phân loại prompt học sinh theo khung APC v4 (24 đặc trưng + 2 prefix bits) "
        "bằng mô hình LLM chạy cục bộ kết hợp với bộ tính toán toán học tĩnh."
    ),
    version="2.0.0",
)

# Cho phép CORS để frontend / client bất kỳ có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Prompt của học sinh cần phân loại")
    student_prompt: Optional[str] = Field(None, description="Alias của 'prompt'")


class ClassifyResponse(BaseModel):
    level: str = Field(..., description="Nhãn phân loại cuối cùng sau khi áp dụng prefix và luật lọc")
    confidence: float = Field(..., description="Độ tin cậy tính bằng Softmax (tau=8)")
    reason: str = Field(..., description="Giải thích lý do phân loại (tương tự như explanation)")
    borderline_with: Optional[str] = Field(None, description="Nhãn phân loại borderline (nếu có)")
    is_coding: bool = Field(..., description="Đặc trưng prefix: Có phải lập trình/coding?")
    has_context: bool = Field(..., description="Đặc trưng prefix: Đủ context và an toàn?")
    predicted_level_math: str = Field(..., description="Nhãn phân loại tính bằng công thức toán (bỏ qua prefix)")
    score: float = Field(..., description="Điểm số cao nhất của nhãn được chọn")
    margin: float = Field(..., description="Khoảng cách độ tin cậy giữa nhãn cao nhất và nhãn cao nhì")
    accept: int = Field(..., description="1: Chấp nhận kết quả (đáp ứng điều kiện biên), 0: Từ chối/Cảnh báo")
    vector: str = Field(..., description="Vector 26-bit nhị phân được trích xuất từ prompt")
    explanation: str = Field(..., description="Giải thích/Lập luận của LLM cho việc gán vector")
    scores: List[float] = Field(..., description="Danh sách điểm số của 6 levels S1..S6")
    gatings: List[bool] = Field(..., description="Trạng thái cổng lọc G1..G6")
    probs: List[float] = Field(..., description="Danh sách xác suất Softmax tương ứng P1..P6")
    latency_seconds: float = Field(..., description="Thời gian suy luận của LLM và xử lý (giây)")
    completion_tokens: int = Field(..., description="Số token đầu ra LLM")
    prompt_tokens: int = Field(..., description="Số token đầu vào LLM")
    total_tokens: int = Field(..., description="Tổng số token đã dùng")
    tokens_per_second: float = Field(..., description="Tốc độ xử lý token/giây")


class ClassifyDebugResponse(ClassifyResponse):
    final_vector_detail: List[dict[str, Any]] = Field(
        ..., description="Vector cuối cùng đưa vào công thức, có tên p1/p2/a1...r10. Hiện tại bằng vector do LLM sinh ra."
    )
    llm_vector_detail: List[dict[str, Any]] = Field(
        ..., description="Vector do LLM/model sinh ra."
    )
    llm_payload: dict[str, Any] = Field(
        default_factory=dict, description="JSON payload thô đã parse từ output của LLM."
    )
    constraint_warnings: List[str] = Field(
        default_factory=list, description="Các cảnh báo constraint của engine."
    )


class HealthResponse(BaseModel):
    api_status: str
    llama_server_status: str
    llama_port: int


class InfoResponse(BaseModel):
    model: str
    llama_port: int
    api_port: int
    prompt_file: Optional[str]
    parallel_slots: int
    gpu_layers: int
    temperature: float
    max_tokens: int
    retry_count: int


class ServerActionRequest(BaseModel):
    action: str = Field(..., description="'start' | 'stop' | 'restart'")


class ServerActionResponse(BaseModel):
    action: str
    result: str


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    """Tự động khởi động llama-server khi API bật lên."""
    print("[app] FastAPI đang khởi động, bật llama-server...")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, llama_manager.start)
    except Exception as exc:
        print(f"[app][WARNING] Không khởi động được llama-server: {exc}")


@app.on_event("shutdown")
async def on_shutdown():
    """Tắt llama-server khi API shutdown để giải phóng VRAM."""
    print("[app] FastAPI đang tắt, dừng llama-server...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, llama_manager.stop)


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/classify", response_model=ClassifyResponse, summary="Phân loại prompt học sinh bằng vector APC v4")
async def classify_prompt(body: ClassifyRequest) -> ClassifyResponse:
    """
    Nhận prompt học sinh và trả về phân loại APC v4 chi tiết.
    - Trích xuất 26 đặc trưng (2 prefix + 24 features) qua mô hình LLM.
    - Chạy bộ suy luận toán học tĩnh (score, gating, softmax, accept) để đưa ra kết quả.
    """
    text = body.prompt or body.student_prompt
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="'prompt' hoặc 'student_prompt' không được để trống.")

    if not llama_manager.healthy():
        raise HTTPException(
            status_code=503,
            detail="llama-server chưa sẵn sàng. Thử lại sau vài giây hoặc kiểm tra trạng thái dịch vụ.",
        )

    loop = asyncio.get_event_loop()
    result, latency, metrics = await loop.run_in_executor(
        None, classifier.classify_with_retry, text.strip()
    )

    return ClassifyResponse(
        level=result["level"],
        confidence=result["confidence"],
        reason=result["explanation"],
        borderline_with=None,
        is_coding=result["is_coding"],
        has_context=result["has_context"],
        predicted_level_math=result["predicted_level_math"],
        score=result["score"],
        margin=result["margin"],
        accept=result["accept"],
        vector=result["vector_str"],
        explanation=result["explanation"],
        scores=result["scores"],
        gatings=result["gatings"],
        probs=result["probs"],
        latency_seconds=round(latency, 3),
        completion_tokens=int(metrics["completion_tokens"]),
        prompt_tokens=int(metrics["prompt_tokens"]),
        total_tokens=int(metrics["total_tokens"]),
        tokens_per_second=float(metrics["tokens_per_second"]),
    )


@app.post("/classify_debug", response_model=ClassifyDebugResponse, summary="Phân loại và trả vector chi tiết")
async def classify_prompt_debug(body: ClassifyRequest) -> ClassifyDebugResponse:
    """Giống /classify nhưng trả thêm chi tiết từng vector p1/p2/a1...r10."""

    text = body.prompt or body.student_prompt
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="'prompt' hoặc 'student_prompt' không được để trống.")

    if not llama_manager.healthy():
        raise HTTPException(
            status_code=503,
            detail="llama-server chưa sẵn sàng. Thử lại sau vài giây hoặc kiểm tra trạng thái dịch vụ.",
        )

    loop = asyncio.get_event_loop()
    result, latency, metrics = await loop.run_in_executor(
        None, classifier.classify_with_retry, text.strip()
    )

    return ClassifyDebugResponse(
        level=result["level"],
        confidence=result["confidence"],
        reason=result["explanation"],
        borderline_with=None,
        is_coding=result["is_coding"],
        has_context=result["has_context"],
        predicted_level_math=result["predicted_level_math"],
        score=result["score"],
        margin=result["margin"],
        accept=result["accept"],
        vector=result["vector_str"],
        explanation=result["explanation"],
        scores=result["scores"],
        gatings=result["gatings"],
        probs=result["probs"],
        latency_seconds=round(latency, 3),
        completion_tokens=int(metrics["completion_tokens"]),
        prompt_tokens=int(metrics["prompt_tokens"]),
        total_tokens=int(metrics["total_tokens"]),
        tokens_per_second=float(metrics["tokens_per_second"]),
        final_vector_detail=vector_detail(result.get("vector_values")),
        llm_vector_detail=vector_detail(result.get("llm_vector_values")),
        llm_payload=result.get("llm_payload", {}),
        constraint_warnings=result.get("constraint_warnings", []),
    )


@app.get("/health", response_model=HealthResponse, summary="Kiểm tra sức khỏe API")
async def health():
    """Trả về trạng thái API và kết nối tới llama-server."""
    llama_ok = llama_manager.healthy()
    return HealthResponse(
        api_status="ok",
        llama_server_status="ok" if llama_ok else "unavailable",
        llama_port=cfg.LLAMA_PORT,
    )


@app.get("/info", response_model=InfoResponse, summary="Thông tin model & cấu hình")
async def info():
    """Trả về thông tin về model, cổng, và các tham số cấu hình đang dùng."""
    return InfoResponse(
        model=cfg.MODEL_NAME,
        llama_port=cfg.LLAMA_PORT,
        api_port=cfg.API_PORT,
        prompt_file=cfg.PROMPT_FILE,
        parallel_slots=cfg.LLAMA_PARALLEL,
        gpu_layers=cfg.LLAMA_NGL,
        temperature=cfg.TEMPERATURE,
        max_tokens=cfg.MAX_TOKENS,
        retry_count=cfg.RETRY_COUNT,
    )


@app.post("/server/action", response_model=ServerActionResponse, summary="Điều khiển llama-server")
async def server_action(body: ServerActionRequest) -> ServerActionResponse:
    """
    Điều khiển thủ công vòng đời của llama-server.exe.
    - `start`   : Khởi động llama-server (nếu chưa chạy)
    - `stop`    : Dừng llama-server (giải phóng VRAM)
    - `restart` : Dừng rồi khởi động lại
    """
    action = body.action.lower().strip()
    loop = asyncio.get_event_loop()

    if action == "start":
        try:
            await loop.run_in_executor(None, llama_manager.start)
            return ServerActionResponse(action=action, result="llama-server đã được khởi động.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    elif action == "stop":
        await loop.run_in_executor(None, llama_manager.stop)
        return ServerActionResponse(action=action, result="Đã gửi lệnh dừng llama-server.")

    elif action == "restart":
        await loop.run_in_executor(None, llama_manager.stop)
        await asyncio.sleep(2)
        try:
            await loop.run_in_executor(None, llama_manager.start)
            return ServerActionResponse(action=action, result="llama-server đã được khởi động lại.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Hành động không hợp lệ: '{action}'. Dùng: start | stop | restart"
        )
