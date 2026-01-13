# app/core/startup.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.model.model_loader import init_model, get_model_info, is_model_available

logger = logging.getLogger("startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 생명주기 관리
    - startup: 모델 로딩
    - shutdown: 리소스 정리
    """
    # Startup
    logger.info("=" * 80)
    logger.info("🚀 애플리케이션 시작")
    logger.info("=" * 80)

    try:
        # 모델 초기화
        logger.info("📦 모델 로딩 시작...")
        success = init_model(force=False)

        if success:
            logger.info("✅ 모델 로딩 성공")
            model_info = get_model_info()
            logger.info(f"   - 모델: {model_info['model_name']}")
            logger.info(f"   - 체크포인트: {model_info['checkpoint']}")
            logger.info(f"   - 디바이스: {model_info['device']}")
            logger.info(f"   - Half Precision: {model_info['half_precision']}")
        else:
            logger.warning("⚠️ 모델 로딩 실패 - 서버는 제한된 기능으로 시작됩니다")
            model_info = get_model_info()
            if model_info.get('error'):
                logger.error(f"   오류: {model_info['error']}")

    except Exception as e:
        logger.error(f"❌ 모델 로딩 중 예외 발생: {e}")
        logger.warning("⚠️ 서버는 제한된 기능으로 시작됩니다")

    logger.info("=" * 80)
    logger.info("✅ 애플리케이션 준비 완료")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("=" * 80)
    logger.info("🛑 애플리케이션 종료 중...")
    logger.info("=" * 80)

    try:
        from app.model.model_loader import cleanup_model
        cleanup_model()
        logger.info("✅ 리소스 정리 완료")
    except Exception as e:
        logger.error(f"❌ 리소스 정리 중 오류: {e}")

    logger.info("=" * 80)
    logger.info("👋 애플리케이션 종료 완료")
    logger.info("=" * 80)


# FastAPI 앱 생성 시 사용
def create_app() -> FastAPI:
    """FastAPI 앱 생성 및 설정"""
    app = FastAPI(
        title="Prompt Purifier API",
        description="KoBART 기반 프롬프트 정제 API",
        version="1.0.0",
        lifespan=lifespan
    )

    # 라우터 등록
    from app.api import purify_router, health_router
    app.include_router(health_router)
    app.include_router(purify_router)

    return app