from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.model.purifier import refine
from app.service.openai_image_service import generate_image
from app.service.translator import translate_to_korean

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


# =========================
# Python → Java 응답 DTO
# =========================
class ImageResponse(BaseModel):
    access_id: str
    is_slang: bool

    original_content: str
    filtered_content: str
    refined_content: str

    image_url: Optional[str] = None
    error_message: Optional[str] = None


@router.post("/image/generate", response_model=ImageResponse)
def generate_image_api(req: ImageRequest):
    logger.info(
        "[IMAGE_API] Request received | access_id=%s, is_slang=%s, promptLen=%d",
        req.access_id, req.is_slang, len(req.original_content)
    )
    filtered_content = ""
    final_prompt = ""
    translated_prompt = ""
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
                image_url=None,
                error_message="프롬프트가 비어있습니다.",
            )

        # -----------------------------
        # 1. 프롬프트 순화(정제) 단계
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
                image_url=None,
                error_message="프롬프트 정제 중 오류가 발생했습니다.",
            )

        # -----------------------------
        # 2. DALL·E-3 이미지 생성
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
                image_url=None,
                error_message="이미지 생성 중 오류가 발생했습니다.",
            )

        # -----------------------------
        # 3. DALL-E 결과 처리
        # -----------------------------
        error_message = dalle_result.get("error_message")

        # 3-1. OpenAI 정책 차단 (정상적인 실패)
        if error_message == "content_policy_violation":
            logger.warning(
                "[IMAGE_API] Content policy violation | access_id=%s, prompt=%s",
                req.access_id, final_prompt[:100]
            )
            try:
                translated_prompt = translate_to_korean(final_prompt)
            except Exception as e:
                logger.error(
                    "[IMAGE_API] Translation failed for policy violation | access_id=%s, error=%s",
                    req.access_id, str(e)
                )
                translated_prompt = final_prompt

            return ImageResponse(
                access_id=req.access_id,
                is_slang=req.is_slang,
                original_content=req.original_content,
                filtered_content=filtered_content,
                refined_content=translate_to_korean(final_prompt),  # 🔁 함께 번역
                image_url=None,
                error_message="콘텐츠 정책에 위반되어 이미지를 만들지 않습니다.",
            )


        # 3-2. 이미지 URL이 없는 경우 (기타 실패)
        # 따로 처리해야 할 부분 : 현재까지 발생한 적 없는 예외
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
                image_url=None,
                error_message=f"이미지 생성 실패: {error_message or '알 수 없는 오류'}",
            )
            # raise HTTPException(
            #     status_code=500,
            #     detail=f"이미지 생성 실패: {dalle_result.get('error_message')}"
            # )

        # -----------------------------
        # 4. Java로 내려줄 최종 응답
        #    (성공한 경우에만 도달)
        # -----------------------------
        try:
            refined_content = dalle_result.get("refined_content") or final_prompt
            translated_prompt = translate_to_korean(refined_content)
            logger.info(
                "[IMAGE_API] Image generated successfully | access_id=%s, imageURL=%s",
                req.access_id, dalle_result.get("image_url")[:50] if dalle_result.get("image_url") else "None"
            )
        except Exception as e:
            logger.error(
                "[IMAGE_API] Translation failed for success case | access_id=%s, error=%s",
                req.access_id, str(e)
            )
            translated_prompt = refined_content if refined_content else final_prompt

        return ImageResponse(
            access_id=req.access_id,
            is_slang=req.is_slang,
            original_content=req.original_content,
            filtered_content=filtered_content,
            refined_content=translated_prompt,
            image_url=dalle_result.get("image_url"),
            error_message=None,
        )
    # -----------------------------
    # 5. 최상위 예외 처리 (모든 예상치 못한 오류)
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
            refined_content=translated_prompt,
            image_url=None,
            error_message="서버 내부 오류가 발생했습니다.",
        )