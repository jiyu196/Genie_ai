# app/service/openai_image_service.py
from app.core.config import OPENAI_API_KEY
from openai import OpenAI, BadRequestError, APIError, APIConnectionError, RateLimitError
import logging
from typing import Any, Dict

logger = logging.getLogger("openai_image_service")
client = OpenAI(api_key=OPENAI_API_KEY)

try:
    if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
        logger.critical("[DALL·E] OpenAI API Key is missing or empty")
        raise ValueError("OpenAI API Key is not configured")

    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("[DALL·E] OpenAI client initialized successfully")
except Exception as e:
    logger.critical("[DALL·E] Failed to initialize OpenAI client | error=%s", str(e))
    client = None


def generate_image(
        models: str = "dall-e-3",
        prompt: str = "",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
) -> Dict[str, Any]:
    """
    ✅ 절대 예외를 밖으로 던지지 않음 (서버가 죽지 않게)
    ✅ 실패 시: image_url=None, refined_content=None, error_message=str(...)
    ✅ 성공 시: image_url=..., refined_content(있으면), error_message=None
    Args:
        models: DALL-E 모델명
        prompt: 이미지 생성 프롬프트
        size: 이미지 크기
        quality: 이미지 품질
        style: 이미지 스타일

    Returns:
        Dict with keys: image_url, refined_content, error_message
    """

    logger.info(
        "[DALL·E] Generate image request | model=%s, size=%s, quality=%s, style=%s, promptLen=%d",
        models, size, quality, style, len(prompt) if prompt else 0
    )
    # -----------------------------
    # 1. 클라이언트 초기화 확인
    # -----------------------------
    if client is None:
        msg = "OpenAI client is not initialized"
        logger.error("[DALL·E] %s", msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": msg,
        }
    # -----------------------------
    # 2. 입력 검증
    # -----------------------------
    if not prompt or not prompt.strip():
        #msg = "prompt is empty"
        msg = "서버 문제로 이미지를 생성하지 못합니다."
        logger.warning("[DALL·E] invalid request: %s", msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": msg,
        }
    if not models or not models.strip():
        #msg = "model name is empty"
        msg = "서버 문제로 이미지를 생성하지 못합니다."
        logger.warning("[DALL·E] Invalid request: %s", msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": msg,
        }

    try:
        # -----------------------------
        # 3. OpenAI API 호출
        # -----------------------------
        logger.debug("[DALL·E] Calling OpenAI API | prompt=%s", prompt[:200])

        resp = client.images.generate(
            model=models,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )
        logger.debug("[DALL·E] OpenAI API call successful")

        # -----------------------------
        # 4. OpenAI 응답 파싱(URL 추출)
        # -----------------------------
        image_url = None
        refined_content = None
        try:
            # 응답 구조 검증
            if not resp or not hasattr(resp, 'data'):
                msg = "Invalid OpenAI response structure (no data)"
                logger.error("[DALL·E] %s | resp=%s", msg, resp)
                return {
                    "image_url": None,
                    "refined_content": None,
                    "error_message": msg,
                }

            if not resp.data or len(resp.data) == 0:
                msg = "OpenAI response data is empty"
                logger.error("[DALL·E] %s", msg)
                return {
                    "image_url": None,
                    "refined_content": None,
                    "error_message": msg,
                }

            # 이미지 URL 추출
            try:
                image_url = resp.data[0].url
                if not image_url or not image_url.strip():
                    logger.warning("[DALL·E] Image URL is empty in response")
                    image_url = None
            except (AttributeError, IndexError, TypeError) as e:
                logger.error("[DALL·E] Failed to extract image URL | error=%s", str(e))
                image_url = None

            # Revised prompt 추출
            if image_url:
                try:
                    refined_content = resp.data[0].revised_prompt
                    if refined_content:
                        logger.info("[DALL·E] Revised prompt extracted | length=%d", len(refined_content))
                except (AttributeError, IndexError, TypeError) as e:
                    logger.debug("[DALL·E] No revised_prompt in response | error=%s", str(e))
                    refined_content = None

        except Exception as e:
            msg = f"Failed to parse OpenAI response: {str(e)}"
            logger.exception("[DALL·E] %s", msg)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": msg,
            }

        # -----------------------------
        # 5. 이미지 URL 검증
        # -----------------------------
        if image_url is None:
            msg = "OpenAI response has no image url"
            logger.error("[DALL·E] %s", msg)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": msg,
            }

        # -----------------------------
        # 6. 성공 응답
        # -----------------------------
        logger.info("[DALL·E] Image generated successfully | url=%s", image_url[:100])
        return {
            "image_url": image_url,
            "refined_content": refined_content,
            "error_message": None,
        }

    except BadRequestError as e:
        msg = str(e)

        # 🔴 OpenAI 정책 차단 (의도된 실패)
        if "content_policy_violation" in msg:
            logger.warning("[DALL·E] blocked by content policy | prompt=%s", prompt[:200])
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "content_policy_violation",
            }

        # 기타 BadRequest 에러
        logger.error(
            "[DALL·E] BadRequestError | model=%s, size=%s, quality=%s, style=%s, prompt=%s, error=%s",
            models, size, quality, style, prompt[:200], msg
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"Bad request: {msg}",
        }

    except RateLimitError as e:
        msg = str(e)
        logger.error(
            "[DALL·E] Rate limit exceeded | model=%s, error=%s",
            models, msg
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": "Rate limit exceeded. Please try again later.",
        }

    except APIConnectionError as e:
        msg = str(e)
        logger.error(
            "[DALL·E] API connection error | model=%s, error=%s",
            models, msg
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": "Failed to connect to OpenAI API. Please check your network.",
        }

    except APIError as e:
        msg = str(e)
        logger.error(
            "[DALL·E] OpenAI API error | model=%s, error=%s",
            models, msg
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"OpenAI API error: {msg}",
        }

    # -----------------------------
    # 8. 예상치 못한 모든 예외 처리
    # -----------------------------
    except Exception as e:
        msg = str(e)
        logger.exception(
            "[DALL·E] Unexpected error | model=%s, size=%s, quality=%s, style=%s, prompt=%s",
            models, size, quality, style, prompt[:200]
        )
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": f"Unexpected error: {msg}",
        }
