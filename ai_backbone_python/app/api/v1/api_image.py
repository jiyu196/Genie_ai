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
# Java → Python 요청 DTO
# =========================
class ImageRequest(BaseModel):
    access_id: str
    original_content: str
    is_slang: bool
    access_id_character: Optional[str] = None  # 캐릭터 설명 (첫 로그인 시에만 전달됨)


# =========================
# Python → Java 응답 DTO
# =========================
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
    logger.info(
        "[IMAGE_API] Request received | access_id=%s, is_slang=%s, has_character=%s, promptLen=%d",
        req.access_id, req.is_slang,
        bool(req.access_id_character),
        len(req.original_content)
    )

    # 변수 초기화
    filtered_content = ""
    final_prompt = ""
    revised_prompt_text = ""
    try:
        # -----------------------------
        # 0. 입력 검증
        # -----------------------------
        if not req.original_content or not req.original_content.strip():
            logger.warning("[IMAGE_API] Empty prompt received | access_id=%s", req.access_id)
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

        # -----------------------------
        # 1. 캐릭터 정보 저장 (새로 받은 경우)
        # -----------------------------
        if req.access_id_character and req.access_id_character.strip():
            character_store.set_character(req.access_id, req.access_id_character)
            logger.info(
                "[IMAGE_API] New character saved | access_id=%s, character_length=%d",
                req.access_id, len(req.access_id_character)
            )

        # 저장된 캐릭터 정보 조회
        saved_character = character_store.get_character(req.access_id)
        if saved_character:
            logger.info(
                "[IMAGE_API] Using saved character | access_id=%s, character=%s",
                req.access_id, saved_character[:100]
            )
        else:
            logger.info(
                "[IMAGE_API] No character found for access_id=%s",
                req.access_id
            )

        # -----------------------------
        # 2. 프롬프트 순화(정제) 단계
        # -----------------------------
        try:
            if req.is_slang:
                logger.info("[IMAGE_API] Refining slang prompt | access_id=%s", req.access_id)
                purified_prompt = refine(req.original_content)

                if not purified_prompt or not purified_prompt.strip():
                    logger.error(
                        "[IMAGE_API] Purifier returned empty result | access_id=%s, original=%s",
                        req.access_id, req.original_content[:100]
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
                logger.info("[IMAGE_API] Prompt refined successfully | access_id=%s", req.access_id)
            else:
                filtered_content = req.original_content
                final_prompt = req.original_content
                logger.info("[IMAGE_API] Using original prompt | access_id=%s", req.access_id)

        except Exception as e:
            logger.exception(
                "[IMAGE_API] Prompt refinement failed | access_id=%s, error=%s",
                req.access_id, str(e)
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

        # -----------------------------
        # 3. 웹툰 스타일 프롬프트 생성 (캐릭터 정보 포함)
        # -----------------------------
        if saved_character:
            # 캐릭터 있을 때: 웹툰 스타일 + 캐릭터 + 장면
            final_prompt = build_webtoon_prompt(
                character_description=saved_character,
                scene_description=final_prompt,
                include_style=True
            )

            # 콘솔에 상세 출력
            log_prompt_construction(
                access_id=req.access_id,
                original_prompt=req.original_content,
                character_description=saved_character,
                final_prompt=final_prompt
            )

            logger.info(
                "[IMAGE_API] Webtoon-style prompt constructed | access_id=%s, final_length=%d",
                req.access_id, len(final_prompt)
            )
        else:
            # 캐릭터 없을 때: 웹툰 스타일 + 장면만
            final_prompt = build_webtoon_prompt(
                character_description=None,
                scene_description=final_prompt,
                include_style=True
            )

            # 콘솔에 상세 출력
            log_prompt_construction(
                access_id=req.access_id,
                original_prompt=req.original_content,
                character_description=None,
                final_prompt=final_prompt
            )

            logger.info(
                "[IMAGE_API] Webtoon-style prompt (no character) | access_id=%s, final_length=%d",
                req.access_id, len(final_prompt)
            )

        # -----------------------------
        # 4. DALL·E-3 이미지 생성
        # -----------------------------
        try:
            logger.info("[IMAGE_API] Calling DALL-E | access_id=%s", req.access_id)
            dalle_result = generate_image(
                models=IMAGE_MODEL,
                prompt=final_prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                style=IMAGE_STYLE,
            )

            if not isinstance(dalle_result, dict):
                logger.error(
                    "[IMAGE_API] Invalid DALL-E response type | access_id=%s, type=%s",
                    req.access_id, type(dalle_result)
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

        except Exception as e:
            logger.exception(
                "[IMAGE_API] DALL-E call failed unexpectedly | access_id=%s, error=%s",
                req.access_id, str(e)
            )
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content="",
                revised_prompt="",
                image_url=None,
                error_message="이미지 생성 중 오류가 발생했습니다.",
            )

        # -----------------------------
        # 5. DALL-E 결과 처리
        # -----------------------------
        error_message = dalle_result.get("error_message")

        # 5-1. OpenAI 정책 차단 (정상적인 실패)
        if error_message == "content_policy_violation":
            logger.warning(
                "[IMAGE_API] Content policy violation | access_id=%s, prompt=%s",
                req.access_id, final_prompt[:100]
            )
            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content=filtered_content,
                revised_prompt="",
                image_url=None,
                error_message="콘텐츠 정책에 위반되어 이미지를 만들지 않습니다.",
            )

        # 5-2. 이미지 URL이 없는 경우 (기타 실패)
        if dalle_result.get("image_url") is None:
            logger.error(
                "[IMAGE_API] No image URL in DALL-E response | access_id=%s, error=%s",
                req.access_id, error_message or "Unknown"
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

        # -----------------------------
        # 6. revised_prompt 처리 및 최종 응답 생성
        # -----------------------------
        dalle_revised_prompt = ""
        try:
            # DALL-E의 revised_prompt 가져오기
            dalle_revised_prompt = dalle_result.get("refined_content") or ""
            if dalle_revised_prompt and dalle_revised_prompt.strip():
                try:
                    logger.info(
                        "[IMAGE_API]+++ Translated successfully | access_id=%s, revised_prompt=%s",
                        req.access_id, dalle_revised_prompt
                    )
                    dalle_revised_prompt = translate_to_korean(dalle_revised_prompt)
                    logger.info(
                        "[IMAGE_API]--- Translated successfully | access_id=%s, revised_prompt=%s",
                        req.access_id, dalle_revised_prompt
                    )
                except Exception as trans_err:
                    logger.error(
                        "[IMAGE_API] Translation failed | access_id=%s, error=%s",
                        req.access_id, str(trans_err)
                    )
                    # 번역 실패 시 원본 한국어 사용
                    dalle_revised_prompt = filtered_content
            else:
                dalle_revised_prompt = filtered_content


            # if dalle_revised_prompt and dalle_revised_prompt.strip():
            #     cleaned_english = remove_style_from_revised_prompt(dalle_revised_prompt)
            #
            #     log_revised_prompt_cleaning(
            #         access_id=req.access_id,
            #         original_revised=dalle_revised_prompt,
            #         cleaned_revised=cleaned_english
            #     )

            # 🔹 실제 사용자 응답용 문장은 여기서 결정
            refined_content_for_response = compose_korean_scene(
                character_description=saved_character,
                scene_description=filtered_content
            )

            logger.info(
                "[IMAGE_API] Refined content (Korean composition) | access_id=%s, refined=%s",
                req.access_id,
                refined_content_for_response
            )

            logger.info(
                "[IMAGE_API] Image generated successfully | access_id=%s, imageURL=%s",
                req.access_id,
                dalle_result.get("image_url")[:50] if dalle_result.get("image_url") else "None"
            )

        except Exception as e:
            logger.error(
                "[IMAGE_API] Error preparing response | access_id=%s, error=%s",
                req.access_id, str(e)
            )
            refined_content_for_response = filtered_content

        # -----------------------------
        # 7. 성공 응답 반환
        # -----------------------------
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

    # -----------------------------
    # 8. 최상위 예외 처리 (모든 예상치 못한 오류)
    # -----------------------------
    except Exception as e:
        logger.exception(
            "[IMAGE_API] Unexpected error in generate_image_api | access_id=%s, error=%s",
            req.access_id if req else "Unknown", str(e)
        )
        return ImageResponse(
            access_id=req.access_id if req else "Unknown",
            is_slang=req.is_slang if req else False,
            original_content=req.original_content if req else "",
            filtered_content=filtered_content,
            refined_content=filtered_content if filtered_content else "",
            revised_prompt="",
            image_url=None,
            error_message="서버 내부 오류가 발생했습니다.",
        )