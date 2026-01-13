# app/api/v1/api_image.py
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, validator
from typing import Optional
import logging

import uuid
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from app.model.purifier import refine, health_check as purifier_health
from app.service.openai_image_service import generate_image
from app.service.translator import translate_to_korean_async
from app.core.character_store import character_store
from app.service.request_trace import trace
from app.service.prompt_builder import (
    build_webtoon_prompt,
    log_prompt_construction,
    compose_korean_scene
)
from app.service.post_processor import post_process

logger = logging.getLogger("api_image")
router = APIRouter()

# =========================
# 설정
# =========================
IMAGE_MODEL = "dall-e-3"
IMAGE_SIZE = "1024x1024"
IMAGE_QUALITY = "standard"
IMAGE_STYLE = "vivid"

# ✅ GPU 작업용 ThreadPool
# - max_workers=3: 여러 요청 동시 처리 가능
# - GPU 추론은 Semaphore로 직렬화
gpu_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gpu_worker")

# 타임아웃 설정
PURIFIER_TIMEOUT = 30
OPENAI_TIMEOUT = 60
TRANSLATOR_TIMEOUT = 10

# ✅ GPU 추론 직렬화용 세마포어 (동시 요청은 받되, GPU inference는 1개씩)
GPU_INFER_SEMAPHORE = asyncio.Semaphore(1)


# =========================
# GPU 추론 함수 (동기)
# =========================
def _run_purifier_sync(text: str, request_id: str) -> str:
    """GPU 추론 (동기) - ThreadPool에서 실행"""
    logger.info(f"[{request_id}] Purifier 시작")
    result = refine(text)
    logger.info(f"[{request_id}] Purifier 완료")
    return result


# =========================
# 비동기 래퍼 함수들
# =========================
async def run_purifier_async(text: str, request_id: str) -> str:
    """GPU 추론 비동기 래퍼"""
    loop = asyncio.get_running_loop()
    try:
        # ✅ GPU 추론은 반드시 1개씩만 실행 (동시성은 API 레벨에서 유지)
        async with GPU_INFER_SEMAPHORE:
            trace("BEFORE MODEL RUN", request_id)
            result = await asyncio.wait_for(
                loop.run_in_executor(gpu_executor, _run_purifier_sync, text, request_id),
                timeout=PURIFIER_TIMEOUT
            )
            return result
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Purifier 타임아웃 ({PURIFIER_TIMEOUT}초)")
        raise TimeoutError(f"프롬프트 정제 시간 초과")


async def run_openai_async(prompt: str, request_id: str) -> dict:
    """OpenAI API 비동기 호출"""
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,  # Default executor (ThreadPool)
                generate_image,
                IMAGE_MODEL,
                prompt,
                IMAGE_SIZE,
                IMAGE_QUALITY,
                IMAGE_STYLE
            ),
            timeout=OPENAI_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] OpenAI 타임아웃 ({OPENAI_TIMEOUT}초)")
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"이미지 생성 시간 초과 ({OPENAI_TIMEOUT}초)"
        }
    except Exception as e:
        logger.exception("[%s] OpenAI 호출 예외", request_id)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"이미지 생성 서비스 오류: {str(e)}",
        }


# =========================
# Request/Response Models
# =========================
class ImageRequest(BaseModel):
    access_id: str = Field(..., min_length=1, max_length=100)
    original_content: str = Field(..., min_length=1, max_length=1000)
    is_slang: bool
    access_id_character: Optional[str] = Field(None, max_length=500)

    @validator('access_id')
    def validate_access_id(cls, v):
        if not v or not v.strip():
            raise ValueError("access_id는 비어있을 수 없습니다")
        return v.strip()

    @validator('original_content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("original_content는 비어있을 수 없습니다")
        return v.strip()

    @validator('access_id_character')
    def validate_character(cls, v):
        if v is not None and v.strip():
            return v.strip()
        return None


class ImageResponse(BaseModel):
    access_id: str
    is_slang: bool
    original_content: str
    filtered_content: str
    refined_content: str
    revised_prompt: Optional[str] = None
    image_url: Optional[str] = None
    error_message: Optional[str] = None


# =========================
# 메인 API 엔드포인트
# =========================
@router.post("/image/generate", response_model=ImageResponse)
async def generate_image_api(req: ImageRequest, request: Request):
    """
    ✅ 이미지 생성 API (최적화된 병렬 처리)

    처리 흐름:
    1. 입력 검증
    2. 캐릭터 정보 저장/조회
    3. [병렬 가능] 프롬프트 정제 (GPU) - Semaphore로 직렬화
    4. 프롬프트 구성
    5. [병렬 가능] OpenAI 이미지 생성
    6. [병렬 가능] 번역
    7. 응답 구성
    """
    # 🔴 요청 단위 추적 ID
    request_id = str(uuid.uuid4())[:8]
    # 🔹 1. 요청이 FastAPI에 도착한 순간
    trace("REQUEST RECEIVED", request_id)

    start_time = time.time()

    logger.info(
        "[%s] === 요청 시작 === | access_id=%s | is_slang=%s | len=%d | client=%s",
        request_id,
        req.access_id,
        req.is_slang,
        len(req.original_content),
        request.client.host if request.client else "unknown"
    )

    filtered_content = req.original_content
    refined_content_for_response = ""
    revised_prompt = ""

    try:
        # =========================
        # 1. 캐릭터 정보 처리
        # =========================
        saved_character = None

        if req.access_id_character:
            character_store.set_character(req.access_id, req.access_id_character)
            trace(f"CHARACTER SAVE user={req.access_id}", request_id)
            logger.info(
                "[%s] 캐릭터 저장 | access_id=%s | len=%d",
                request_id, req.access_id, len(req.access_id_character)
            )
        else:
            trace(f"CHARACTER LOAD user={req.access_id}", request_id)

        saved_character = character_store.get_character(req.access_id)
        if saved_character:
            logger.info(
                "[%s] 캐릭터 조회 성공 | preview=%s...",
                request_id, saved_character[:50]
            )

        # =========================
        # 2. 프롬프트 정제 (선택적)
        # =========================
        if req.is_slang:
            logger.info(f"[{request_id}] 프롬프트 정제 시작")

            try:
                # GPU 추론 (비동기 + Semaphore로 직렬화)
                purified = await run_purifier_async(req.original_content, request_id)

                # 후처리 (CPU, 빠름)
                purified = post_process(req.original_content, purified)

                if not purified or not purified.strip():
                    logger.error(f"[{request_id}] 정제 결과 비어있음")
                    return ImageResponse(
                        access_id=req.access_id,
                        is_slang=req.is_slang,
                        original_content=req.original_content,
                        filtered_content="",
                        refined_content="",
                        error_message="프롬프트 정제 실패"
                    )

                filtered_content = purified
                logger.info(
                    "[%s] 정제 완료 | before_len=%d | after_len=%d",
                    request_id, len(req.original_content), len(purified)
                )

            except TimeoutError as e:
                logger.error(f"[{request_id}] 정제 타임아웃")
                return ImageResponse(
                    access_id=req.access_id,
                    is_slang=req.is_slang,
                    original_content=req.original_content,
                    filtered_content=req.original_content,
                    refined_content="",
                    error_message=str(e)
                )

            except Exception as e:
                logger.error(f"[{request_id}] 정제 오류: {e}")
                filtered_content = req.original_content

        else:
            logger.info(f"[{request_id}] 정제 스킵 (is_slang=False)")

        # =========================
        # 3. 웹툰 스타일 프롬프트 구성
        # =========================
        try:
            final_prompt = build_webtoon_prompt(
                character_description=saved_character,
                scene_description=filtered_content,
                include_style=True
            )

            log_prompt_construction(
                access_id=req.access_id,
                original_prompt=req.original_content,
                character_description=saved_character,
                final_prompt=final_prompt
            )

            logger.info(
                "[%s] 프롬프트 구성 완료 | final_len=%d",
                request_id, len(final_prompt)
            )

        except Exception as e:
            logger.error(f"[{request_id}] 프롬프트 구성 실패: {e}")
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                error_message="프롬프트 구성 오류"
            )

        # =========================
        # 4. OpenAI 이미지 생성 (비동기)
        # =========================
        logger.info(f"[{request_id}] OpenAI 호출 시작")

        dalle_result = await run_openai_async(final_prompt, request_id)

        # 결과 검증
        if not isinstance(dalle_result, dict):
            logger.error(f"[{request_id}] OpenAI 응답 타입 오류")
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                error_message="이미지 생성 서비스 오류"
            )

        # 콘텐츠 정책 위반
        if dalle_result.get("error_message") == "content_policy_violation":
            logger.warning(f"[{request_id}] 콘텐츠 정책 위반")
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content=filtered_content,
                error_message="콘텐츠 정책에 위반되어 이미지를 생성할 수 없습니다."
            )

        # 이미지 URL 없음
        if not dalle_result.get("image_url"):
            error_msg = dalle_result.get("error_message") or "알 수 없는 오류"
            logger.error(f"[{request_id}] 이미지 생성 실패: {error_msg}")
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                error_message=f"이미지 생성 실패: {error_msg}"
            )

        logger.info(f"[{request_id}] 이미지 생성 성공")

        # =========================
        # 5. revised_prompt 번역 (비동기)
        # =========================
        dalle_revised = dalle_result.get("refined_content", "").strip()

        if dalle_revised:
            logger.info(f"[{request_id}] 번역 시작")

            try:
                revised_prompt = await asyncio.wait_for(
                    translate_to_korean_async(dalle_revised),
                    timeout=TRANSLATOR_TIMEOUT
                )
                logger.info(f"[{request_id}] 번역 완료")

            except asyncio.TimeoutError:
                logger.warning(f"[{request_id}] 번역 타임아웃, 원본 사용")
                revised_prompt = filtered_content

            except Exception as e:
                logger.error(f"[{request_id}] 번역 실패: {e}")
                revised_prompt = filtered_content
        else:
            revised_prompt = filtered_content

        # =========================
        # 6. 최종 응답용 한국어 문장 구성
        # =========================
        try:
            refined_content_for_response = compose_korean_scene(
                character_description=saved_character,
                scene_description=filtered_content
            )
        except Exception as e:
            logger.error(f"[{request_id}] 응답 구성 실패: {e}")
            refined_content_for_response = filtered_content

        # =========================
        # 7. 성공 응답
        # =========================
        processing_time = time.time() - start_time

        logger.info(
            "[%s] === 요청 완료 (성공) === | time=%.2fs | url=%s...",
            request_id, processing_time, dalle_result["image_url"][:60]
        )

        return ImageResponse(
            access_id=req.access_id,
            is_slang=req.is_slang,
            original_content=req.original_content,
            filtered_content=filtered_content,
            refined_content=refined_content_for_response,
            revised_prompt=revised_prompt,
            image_url=dalle_result["image_url"],
            error_message=None
        )

    # =========================
    # 8. 전역 예외 처리
    # =========================
    except Exception as e:
        processing_time = time.time() - start_time

        logger.exception(
            "[%s] === 요청 완료 (오류) === | time=%.2fs",
            request_id, processing_time
        )

        return ImageResponse(
            access_id=req.access_id,
            is_slang=req.is_slang,
            original_content=req.original_content,
            filtered_content=filtered_content,
            refined_content=refined_content_for_response or filtered_content,
            revised_prompt=revised_prompt,
            error_message=f"서버 오류: {str(e)}"
        )


# =========================
# 헬스체크 엔드포인트
# =========================
@router.get("/image/health")
async def health_check():
    """서비스 상태 확인"""
    from app.service.openai_image_service import health_check as openai_health

    purifier_status = purifier_health()
    openai_status = openai_health()
    character_stats = character_store.get_stats()

    is_healthy = (
            purifier_status.get("model_available", False) and
            openai_status.get("client_initialized", False)
    )

    return {
        "status": "healthy" if is_healthy else "degraded",
        "model": IMAGE_MODEL,
        "purifier": purifier_status,
        "openai": openai_status,
        "character_store": character_stats,
        "gpu_executor": {
            "max_workers": gpu_executor._max_workers,
        },
        "gpu_semaphore": {
            "limit": GPU_INFER_SEMAPHORE._value,
            "note": "GPU 추론은 동시에 1개만 실행"
        },
        "timeouts": {
            "purifier": PURIFIER_TIMEOUT,
            "openai": OPENAI_TIMEOUT,
            "translator": TRANSLATOR_TIMEOUT
        }
    }


@router.get("/image/info")
async def service_info():
    """서비스 정보"""
    return {
        "service": "Webtoon AI Image Generator",
        "version": "2.0.0",
        "model": IMAGE_MODEL,
        "image_size": IMAGE_SIZE,
        "features": [
            "프롬프트 정제 (KoBART)",
            "캐릭터 일관성 유지",
            "웹툰 스타일 생성",
            "자동 번역",
            "GPU 추론 직렬화"
        ]
    }