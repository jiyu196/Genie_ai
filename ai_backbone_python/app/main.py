# app/main.py
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.api.v1.api_image import router as image_router
from app.model.model_loader import init_model
#from app.core.config import validate_config, print_config
import logging
import time
import sys
import traceback

# =========================
# 로깅 설정
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ai_backbone_python.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("main")

# =========================
# FastAPI 앱 생성
# =========================
app = FastAPI(
    title="Webtoon AI Backbone",
    version="1.0.0",
    description="이미지 생성 및 프롬프트 정제 API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =========================
# CORS 설정
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)


# =========================
# 미들웨어: 요청 추적 및 로깅
# =========================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    ✅ 모든 요청 로깅 및 처리 시간 측정
    ✅ 느린 요청 경고
    ✅ 요청 ID 추가
    """
    # 요청 ID 생성 (이미 있으면 사용)
    request_id = request.headers.get("X-Request-ID", str(time.time_ns())[:16])

    # 시작 시간
    start_time = time.time()

    # 요청 로깅
    logger.info(
        "[%s] --> %s %s | client=%s | user-agent=%s",
        request_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        request.headers.get("user-agent", "unknown")[:50]
    )

    try:
        # 요청 처리
        response = await call_next(request)

        # 처리 시간 계산
        process_time = time.time() - start_time

        # 응답 로깅
        log_level = logging.WARNING if process_time > 10 else logging.INFO
        logger.log(
            log_level,
            "[%s] <-- %s %s | status=%d | time=%.2fs",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            process_time
        )

        # 응답 헤더에 메타데이터 추가
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        # 미들웨어 레벨 예외 처리
        process_time = time.time() - start_time
        logger.exception(
            "[%s] <-- %s %s | ERROR | time=%.2fs",
            request_id,
            request.method,
            request.url.path,
            process_time
        )

        # 500 에러 반환
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "서버 오류가 발생했습니다",
                "request_id": request_id,
                "path": request.url.path
            },
            headers={
                "X-Process-Time": f"{process_time:.3f}",
                "X-Request-ID": request_id
            }
        )


# =========================
# 예외 핸들러: 입력 검증 오류
# =========================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    ✅ Pydantic 검증 오류를 사용자 친화적으로 변환
    """
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")

    logger.warning(
        "[VALIDATION] 입력 검증 실패 | path=%s | errors=%s",
        request.url.path,
        errors
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "입력 데이터가 올바르지 않습니다",
            "details": errors,
            "path": request.url.path
        }
    )


# =========================
# 예외 핸들러: 전역 오류
# =========================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    ✅ 예상치 못한 모든 오류 처리
    ✅ 서버 크래시 방지
    """
    # 상세 에러 로깅
    logger.error("=" * 80)
    logger.error("[GLOBAL ERROR] 예상치 못한 오류 발생")
    logger.error(f"Path: {request.url.path}")
    logger.error(f"Method: {request.method}")
    logger.error(f"Error Type: {type(exc).__name__}")
    logger.error(f"Error Message: {str(exc)}")
    logger.error("Traceback:")
    logger.error(traceback.format_exc())
    logger.error("=" * 80)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버 오류가 발생했습니다. 관리자에게 문의하세요.",
            "error_type": type(exc).__name__,
            "path": request.url.path
        }
    )


# =========================
# 루트 엔드포인트
# =========================
@app.get("/")
async def root():
    """API 루트"""
    return {
        "service": "Webtoon AI Backbone",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# =========================
# 헬스체크 엔드포인트
# =========================
@app.get("/health")
async def health():
    """전체 서비스 상태 확인"""
    from app.model.model_loader import get_model_info
    from app.service.openai_image_service import health_check as openai_health

    model_info = get_model_info()
    openai_status = openai_health()

    is_healthy = (
            model_info.get("loaded", False) and
            openai_status.get("client_initialized", False)
    )

    return {
        "status": "healthy" if is_healthy else "degraded",
        "version": "1.0.0",
        "service": "Webtoon AI Backbone",
        "components": {
            "model": model_info,
            "openai": openai_status
        }
    }


# =========================
# 라우터 등록
# =========================
app.include_router(
    image_router,
    prefix="/api/v1",
    tags=["image"]
)


# =========================
# 시작 이벤트
# =========================
@app.on_event("startup")
async def startup_event():
    """
    ✅ 서버 시작 시 초기화
    ✅ 모델 로딩
    ✅ 설정 검증
    ✅ 라우트 출력
    """
    logger.info("\n" + "=" * 80)
    logger.info("🚀 Webtoon AI Backbone 서버 시작")
    logger.info("=" * 80)

    # 1. 모델 초기화 (✅ 추가됨)
    logger.info("\n📥 모델 로딩 중...")
    try:
        if init_model():
            logger.info("✅ 모델 초기화 성공")
        else:
            logger.error("❌ 모델 초기화 실패")
            logger.error("   ⚠️ 프롬프트 정제 기능이 비활성화됩니다")
            logger.error("   ⚠️ is_slang=true 요청은 정제 없이 처리됩니다")
    except Exception as e:
        logger.error(f"❌ 모델 초기화 중 예외 발생: {e}")
        logger.error("   ⚠️ 서버는 시작되지만 정제 기능이 작동하지 않습니다")

    # 2. 설정 정보 출력 (주석 처리된 부분 - 필요시 활성화)
    # print_config()

    # 3. 설정 검증 (주석 처리된 부분 - 필요시 활성화)
    # logger.info("\n🔍 설정 검증 중...")
    # config_valid = validate_config()
    #
    # if not config_valid:
    #     logger.error("⚠️⚠️⚠️ 설정 검증 실패 ⚠️⚠️⚠️")
    #     logger.error("서버가 정상적으로 작동하지 않을 수 있습니다")

    # 4. 라우트 출력
    logger.info("\n📋 등록된 라우트:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods)
            logger.info(f"  [{methods:6s}] {route.path}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ 서버 시작 완료")
    logger.info("=" * 80 + "\n")


# =========================
# 종료 이벤트
# =========================
@app.on_event("shutdown")
async def shutdown_event():
    """
    ✅ 서버 종료 시 정리 작업
    """
    logger.info("\n" + "=" * 80)
    logger.info("🛑 서버 종료 중...")
    logger.info("=" * 80)

    # 1. GPU 메모리 정리
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ GPU 메모리 정리 완료")
    except Exception as e:
        logger.warning(f"GPU 메모리 정리 실패: {e}")

    # 2. ThreadPoolExecutor 정리
    try:
        from app.api.v1.api_image import gpu_executor
        gpu_executor.shutdown(wait=True, cancel_futures=True)
        logger.info("✅ ThreadPoolExecutor 종료 완료")
    except Exception as e:
        logger.warning(f"ThreadPoolExecutor 종료 실패: {e}")

    logger.info("=" * 80)
    logger.info("👋 서버 종료 완료")
    logger.info("=" * 80 + "\n")


# =========================
# 개발용: 직접 실행
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 시에만 True
        log_level="info"
    )