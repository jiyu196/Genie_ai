# app/service/openai_image_service.py
from app.core.config import OPENAI_API_KEY
from openai import OpenAI, BadRequestError, APIError, APIConnectionError, RateLimitError
import logging
from typing import Any, Dict

logger = logging.getLogger("openai_image_service")

# =========================
# OpenAI 클라이언트 초기화
# =========================
client = None

try:
    if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
        logger.critical("[INIT] OpenAI API Key 누락 또는 비어있음")
        raise ValueError("OpenAI API Key가 설정되지 않았습니다")

    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("[INIT] OpenAI 클라이언트 초기화 성공")

except Exception as e:
    logger.critical("[INIT] OpenAI 클라이언트 초기화 실패 | error=%s", str(e))
    client = None


def generate_image(
        models: str = "dall-e-3",
        prompt: str = "",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
) -> Dict[str, Any]:
    """
    DALL-E 이미지 생성 함수

    ✅ 절대 예외를 밖으로 던지지 않음 (서버 안정성 보장)
    ✅ 실패 시: image_url=None, refined_content=None, error_message=str
    ✅ 성공 시: image_url=str, refined_content=str/None, error_message=None

    Args:
        models: DALL-E 모델명 (기본값: dall-e-3)
        prompt: 이미지 생성 프롬프트
        size: 이미지 크기 (기본값: 1024x1024)
        quality: 이미지 품질 (기본값: standard)
        style: 이미지 스타일 (기본값: vivid)

    Returns:
        Dict[str, Any]: {
            "image_url": str | None,
            "refined_content": str | None,  # OpenAI의 revised_prompt
            "error_message": str | None
        }
    """

    logger.info(
        "[REQUEST] 이미지 생성 요청 | model=%s | size=%s | quality=%s | style=%s | prompt_len=%d",
        models, size, quality, style, len(prompt) if prompt else 0
    )

    # =========================
    # 1. 클라이언트 초기화 확인
    # =========================
    if client is None:
        error_msg = "OpenAI 클라이언트가 초기화되지 않았습니다"
        logger.error("[VALIDATION] %s", error_msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": error_msg,
        }

    # =========================
    # 2. 입력 검증
    # =========================
    if not prompt or not prompt.strip():
        error_msg = "프롬프트가 비어있습니다"
        logger.warning("[VALIDATION] %s", error_msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": error_msg,
        }

    if not models or not models.strip():
        error_msg = "모델명이 비어있습니다"
        logger.warning("[VALIDATION] %s", error_msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": error_msg,
        }

    try:
        # =========================
        # 3. OpenAI API 호출
        # =========================
        logger.info("[API_CALL] DALL-E API 호출 시작 | prompt=%s...", prompt[:100])

        resp = client.images.generate(
            model=models,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )

        logger.info("[API_CALL] DALL-E API 호출 성공")

        # =========================
        # 4. 응답 구조 검증
        # =========================
        if not resp:
            error_msg = "OpenAI 응답이 비어있습니다"
            logger.error("[PARSE] %s", error_msg)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": error_msg,
            }

        if not hasattr(resp, 'data'):
            error_msg = "OpenAI 응답 구조가 잘못되었습니다 (data 속성 없음)"
            logger.error("[PARSE] %s | resp_type=%s", error_msg, type(resp).__name__)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": error_msg,
            }

        if not resp.data or len(resp.data) == 0:
            error_msg = "OpenAI 응답 데이터가 비어있습니다"
            logger.error("[PARSE] %s", error_msg)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": error_msg,
            }

        # =========================
        # 5. 이미지 URL 추출
        # =========================
        try:
            image_url = resp.data[0].url

            if not image_url or not image_url.strip():
                error_msg = "이미지 URL이 비어있습니다"
                logger.error("[PARSE] %s", error_msg)
                return {
                    "image_url": None,
                    "refined_content": None,
                    "error_message": error_msg,
                }

            logger.info("[PARSE] 이미지 URL 추출 성공 | url=%s...", image_url[:60])

        except (AttributeError, IndexError, TypeError) as e:
            error_msg = f"이미지 URL 추출 실패: {str(e)}"
            logger.error("[PARSE] %s", error_msg)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": error_msg,
            }

        # =========================
        # 6. Revised Prompt 추출 (선택적)
        # =========================
        refined_content = None

        try:
            item = resp.data[0]

            # dict 또는 객체 모두 대응
            if isinstance(item, dict):
                refined_content = item.get("revised_prompt")
            else:
                refined_content = getattr(item, "revised_prompt", None)

            if refined_content and refined_content.strip():
                logger.info("[PARSE] Revised prompt 추출 성공 | length=%d", len(refined_content))
            else:
                logger.debug("[PARSE] Revised prompt 없음 또는 비어있음")
                refined_content = None

        except Exception as e:
            logger.debug("[PARSE] Revised prompt 추출 실패 (무시 가능) | error=%s", str(e))
            refined_content = None

        # =========================
        # 7. 성공 응답 반환
        # =========================
        logger.info("[SUCCESS] 이미지 생성 완료 | url=%s...", image_url[:60])
        return {
            "image_url": image_url,
            "refined_content": refined_content,
            "error_message": None,
        }

    # =========================
    # 8. OpenAI 예외 처리
    # =========================
    except BadRequestError as e:
        error_detail = str(e)

        # 콘텐츠 정책 위반 (의도된 실패)
        if "content_policy_violation" in error_detail.lower():
            logger.warning(
                "[POLICY] 콘텐츠 정책 위반 | prompt=%s...",
                prompt[:100]
            )
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "콘텐츠  정책에 위반되어 이미지를 만들지 않습니다.",
            }

        # 기타 잘못된 요청
        logger.error(
            "[ERROR] BadRequest 오류 | model=%s | size=%s | quality=%s | style=%s | error=%s",
            models, size, quality, style, error_detail
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"잘못된 요청: {error_detail}",
        }

    except RateLimitError as e:
        error_detail = str(e)
        logger.error(
            "[ERROR] API 사용량 초과 | model=%s | error=%s",
            models, error_detail
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": "API 사용량을 초과했습니다. 잠시 후 다시 시도해주세요.",
        }

    except APIConnectionError as e:
        error_detail = str(e)
        logger.error(
            "[ERROR] API 연결 실패 | model=%s | error=%s",
            models, error_detail
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": "OpenAI API에 연결할 수 없습니다. 네트워크를 확인해주세요.",
        }

    except APIError as e:
        error_detail = str(e)
        logger.error(
            "[ERROR] OpenAI API 오류 | model=%s | error=%s",
            models, error_detail
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"OpenAI API 오류: {error_detail}",
        }

    # =========================
    # 9. 예상치 못한 모든 예외 처리
    # =========================
    except Exception as e:
        error_detail = str(e)
        logger.exception(
            "[ERROR] 예상치 못한 오류 | model=%s | size=%s | quality=%s | style=%s",
            models, size, quality, style
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"예상치 못한 오류가 발생했습니다: {error_detail}",
        }