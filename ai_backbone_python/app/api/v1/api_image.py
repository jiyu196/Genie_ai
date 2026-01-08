from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.model.purifier import refine
from app.service.openai_image_service import generate_image
from app.service.translator import translate_to_korean
from app.core.character_store import character_store
from app.service.prompt_builder import (
    compose_korean_scene,
    build_webtoon_prompt,
    log_prompt_construction,
    remove_style_from_revised_prompt,
    log_revised_prompt_cleaning
)

logger = logging.getLogger("api_image")
router = APIRouter()

IMAGE_MODEL = "dall-e-3"
IMAGE_SIZE = "1024x1024"
IMAGE_QUALITY = "standard"
IMAGE_STYLE = "vivid"


# =========================
# Request/Response Models
# =========================
class ImageRequest(BaseModel):
    access_id: str
    original_content: str
    is_slang: bool
    access_id_character: Optional[str] = None


class ImageResponse(BaseModel):
    access_id: str
    is_slang: bool
    original_content: str
    filtered_content: str
    refined_content: str
    revised_prompt: str
    image_url: Optional[str] = None
    error_message: Optional[str] = None


@router.post("/image/generate", response_model=ImageResponse)
def generate_image_api(req: ImageRequest):
    """
    이미지 생성 API 엔드포인트
    - 욕설/비속어 필터링
    - 웹툰 스타일 프롬프트 생성
    - DALL-E 이미지 생성
    """

    # 🔹 변수 초기화 (try 블록 밖에서 초기화하여 except에서도 안전하게 사용)
    filtered_content = ""
    final_prompt = ""
    dalle_revised_prompt = ""
    refined_content_for_response = ""
    saved_character = None

    logger.info(
        "[REQUEST] 이미지 생성 요청 수신 | access_id=%s | is_slang=%s | has_character=%s | prompt_len=%d",
        req.access_id,
        req.is_slang,
        bool(req.access_id_character),
        len(req.original_content)
    )

    try:
        # =========================
        # 1. 입력 검증
        # =========================
        if not req.original_content or not req.original_content.strip():
            logger.warning("[VALIDATION] 빈 프롬프트 수신 | access_id=%s", req.access_id)
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content="",
                refined_content="",
                revised_prompt="",
                image_url=None,
                error_message="프롬프트가 비어있습니다.",
            )

        # =========================
        # 2. 캐릭터 정보 저장 및 조회
        # =========================
        if req.access_id_character and req.access_id_character.strip():
            character_store.set_character(req.access_id, req.access_id_character)
            logger.info(
                "[CHARACTER] 새 캐릭터 저장 완료 | access_id=%s | char_len=%d",
                req.access_id,
                len(req.access_id_character)
            )

        saved_character = character_store.get_character(req.access_id)
        if saved_character:
            logger.info(
                "[CHARACTER] 저장된 캐릭터 사용 | access_id=%s | char_preview=%s...",
                req.access_id,
                saved_character[:50]
            )
        else:
            logger.info("[CHARACTER] 캐릭터 정보 없음 | access_id=%s", req.access_id)

        # =========================
        # 3. 프롬프트 필터링 (욕설/비속어 제거)
        # =========================
        if req.is_slang:
            logger.info("[FILTER] 욕설 필터링 시작 | access_id=%s", req.access_id)

            try:
                purified_prompt = refine(req.original_content)

                if not purified_prompt or not purified_prompt.strip():
                    logger.error(
                        "[FILTER] 필터링 결과 빈 문자열 반환 | access_id=%s | original=%s...",
                        req.access_id,
                        req.original_content[:50]
                    )
                    return ImageResponse(
                        access_id=req.access_id,
                        is_slang=req.is_slang,
                        original_content=req.original_content,
                        filtered_content="",
                        refined_content="",
                        revised_prompt="",
                        image_url=None,
                        error_message="프롬프트 정제 중 오류가 발생했습니다.",
                    )

                filtered_content = purified_prompt
                final_prompt = purified_prompt
                logger.info(
                    "[FILTER] 필터링 완료 | access_id=%s | filtered=%s...",
                    req.access_id,
                    filtered_content[:50]
                )
            except Exception as filter_err:
                logger.error(
                    "[FILTER] 필터링 함수 예외 발생 | access_id=%s | error=%s",
                    req.access_id,
                    str(filter_err)
                )
                return ImageResponse(
                    access_id=req.access_id,
                    is_slang=req.is_slang,
                    original_content=req.original_content,
                    filtered_content="",
                    refined_content="",
                    revised_prompt="",
                    image_url=None,
                    error_message="프롬프트 정제 중 오류가 발생했습니다.",
                )
        else:
            filtered_content = req.original_content
            final_prompt = req.original_content
            logger.info("[FILTER] 필터링 스킵 (is_slang=False) | access_id=%s", req.access_id)

        # =========================
        # 4. 웹툰 스타일 프롬프트 생성
        # =========================
        try:
            final_prompt = build_webtoon_prompt(
                character_description=saved_character,
                scene_description=final_prompt,
                include_style=True
            )

            log_prompt_construction(
                access_id=req.access_id,
                original_prompt=req.original_content,
                character_description=saved_character,
                final_prompt=final_prompt
            )

            logger.info(
                "[PROMPT] 웹툰 스타일 프롬프트 생성 완료 | access_id=%s | has_character=%s | final_len=%d",
                req.access_id,
                bool(saved_character),
                len(final_prompt)
            )
        except Exception as prompt_err:
            logger.error(
                "[PROMPT] 프롬프트 생성 실패 | access_id=%s | error=%s",
                req.access_id,
                str(prompt_err)
            )
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                revised_prompt="",
                image_url=None,
                error_message="프롬프트 생성 중 오류가 발생했습니다.",
            )

        # =========================
        # 5. DALL-E 이미지 생성
        # =========================
        logger.info("[DALLE] 이미지 생성 API 호출 | access_id=%s", req.access_id)

        try:
            dalle_result = generate_image(
                models=IMAGE_MODEL,
                prompt=final_prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                style=IMAGE_STYLE,
            )
        except Exception as dalle_err:
            logger.error(
                "[DALLE] API 호출 실패 | access_id=%s | error=%s",
                req.access_id,
                str(dalle_err)
            )
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                revised_prompt="",
                image_url=None,
                error_message="이미지 생성 API 호출 중 오류가 발생했습니다.",
            )

        # 🔹 결과 타입 검증
        if not isinstance(dalle_result, dict):
            logger.error(
                "[DALLE] 잘못된 응답 타입 | access_id=%s | type=%s",
                req.access_id,
                type(dalle_result).__name__
            )
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                revised_prompt="",
                image_url=None,
                error_message="이미지 생성 서비스 오류가 발생했습니다.",
            )

        # =========================
        # 6. DALL-E 결과 처리
        # =========================
        error_message = dalle_result.get("error_message")

        # 6-1. 콘텐츠 정책 위반
        if error_message == "content_policy_violation":
            logger.warning(
                "[DALLE] 콘텐츠 정책 위반 | access_id=%s | prompt=%s...",
                req.access_id,
                final_prompt[:50]
            )
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content=filtered_content,
                revised_prompt="",
                image_url=None,
                error_message="콘텐츠 정책에 위반되어 이미지를 생성할 수 없습니다.",
            )

        # 6-2. 이미지 URL 없음 (기타 오류)
        if not dalle_result.get("image_url"):
            logger.error(
                "[DALLE] 이미지 URL 없음 | access_id=%s | error=%s",
                req.access_id,
                error_message or "Unknown"
            )
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                revised_prompt="",
                image_url=None,
                error_message=f"이미지 생성 실패: {error_message or '알 수 없는 오류'}",
            )

        # =========================
        # 7. revised_prompt 처리 및 한글 번역
        # =========================
        dalle_revised_prompt = dalle_result.get("refined_content", "").strip()

        if dalle_revised_prompt:
            logger.info(
                "[TRANSLATE] 번역 시작 | access_id=%s | original=%s...",
                req.access_id,
                dalle_revised_prompt[:50]
            )

            try:
                translated_prompt = translate_to_korean(dalle_revised_prompt)
                dalle_revised_prompt = translated_prompt
                logger.info(
                    "[TRANSLATE] 번역 완료 | access_id=%s | translated=%s...",
                    req.access_id,
                    translated_prompt[:50]
                )
            except Exception as trans_err:
                logger.error(
                    "[TRANSLATE] 번역 실패, 필터링된 원본 사용 | access_id=%s | error=%s",
                    req.access_id,
                    str(trans_err)
                )
                dalle_revised_prompt = filtered_content
        else:
            # 🔹 revised_prompt가 없으면 필터링된 원본 사용
            dalle_revised_prompt = filtered_content
            logger.info(
                "[TRANSLATE] revised_prompt 없음, 필터링된 원본 사용 | access_id=%s",
                req.access_id
            )

        # =========================
        # 8. 최종 응답용 한국어 문장 생성
        # =========================
        try:
            refined_content_for_response = compose_korean_scene(
                character_description=saved_character,
                scene_description=filtered_content
            )

            logger.info(
                "[RESPONSE] 최종 응답 생성 완료 | access_id=%s | refined=%s...",
                req.access_id,
                refined_content_for_response[:50]
            )
        except Exception as compose_err:
            logger.error(
                "[RESPONSE] 응답 생성 실패, 필터링된 원본 사용 | access_id=%s | error=%s",
                req.access_id,
                str(compose_err)
            )
            refined_content_for_response = filtered_content

        logger.info(
            "[SUCCESS] 이미지 생성 성공 | access_id=%s | image_url=%s...",
            req.access_id,
            dalle_result.get("image_url", "")[:60]
        )

        # =========================
        # 9. 성공 응답 반환
        # =========================
        return ImageResponse(
            access_id=req.access_id,
            is_slang=req.is_slang,
            original_content=req.original_content,
            filtered_content=filtered_content,
            refined_content=refined_content_for_response,
            revised_prompt=dalle_revised_prompt,
            image_url=dalle_result.get("image_url"),
            error_message=None,
        )

    except Exception as e:
        logger.exception(
            "[ERROR] 예상치 못한 오류 발생 | access_id=%s | error=%s",
            req.access_id,
            str(e)
        )

        # 🔹 에러 발생 시에도 가능한 정보 반환
        return ImageResponse(
            access_id=req.access_id,
            is_slang=req.is_slang,
            original_content=req.original_content,
            filtered_content=filtered_content,
            refined_content=refined_content_for_response or filtered_content,
            revised_prompt=dalle_revised_prompt,
            image_url=None,
            error_message="서버 내부 오류가 발생했습니다.",
        )