# app/service/openai_image_service.py
from app.core.config import OPENAI_API_KEY
from openai import (
    OpenAI, OpenAIError, BadRequestError, APIError,
    APIConnectionError, RateLimitError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, NotFoundError,
    UnprocessableEntityError, InternalServerError, ConflictError
)
import logging
import time
from typing import Any, Dict, Optional
import json

logger = logging.getLogger("openai_image_service")

# =========================
# 상수 정의
# =========================
MAX_PROMPT_LENGTH = 4000  # DALL-E-3 제한
MIN_PROMPT_LENGTH = 1
SUPPORTED_MODELS = ["dall-e-2", "dall-e-3"]
SUPPORTED_SIZES = {
    "dall-e-2": ["256x256", "512x512", "1024x1024"],
    "dall-e-3": ["1024x1024", "1024x1792", "1792x1024"]
}
SUPPORTED_QUALITIES = ["standard", "hd"]
SUPPORTED_STYLES = ["vivid", "natural"]

# 재시도 설정
MAX_RETRIES = 3
RETRY_DELAY = 2  # 초

# =========================
# OpenAI 클라이언트 초기화
# =========================
client = None
client_init_error = None

try:
    # API 키 검증
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다")

    if not OPENAI_API_KEY.strip():
        raise ValueError("OPENAI_API_KEY가 비어있습니다")

    # API 키 형식 검증 (유연하게)
    api_key_stripped = OPENAI_API_KEY.strip()
    if len(api_key_stripped) < 20:
        raise ValueError("OPENAI_API_KEY가 너무 짧습니다 (최소 20자)")

    if not api_key_stripped.startswith("sk-"):
        logger.warning("⚠️ OpenAI API 키가 'sk-'로 시작하지 않습니다. 형식을 확인하세요.")

    # 클라이언트 초기화
    client = OpenAI(
        api_key=api_key_stripped,
        timeout=90.0,  # 전체 타임아웃 90초
        max_retries=0  # 수동 재시도 제어
    )

    logger.info("✅ OpenAI 클라이언트 초기화 성공")

except ValueError as e:
    client_init_error = str(e)
    logger.critical(f"❌ OpenAI 클라이언트 초기화 실패: {client_init_error}")
    client = None

except Exception as e:
    client_init_error = f"예상치 못한 오류: {str(e)}"
    logger.critical(f"❌ OpenAI 클라이언트 초기화 실패: {client_init_error}")
    logger.exception("상세 오류:")
    client = None


# =========================
# 유틸리티 함수
# =========================
def sanitize_prompt(prompt: str) -> str:
    """
    ✅ 프롬프트 정제 (특수문자, 공백 등)
    """
    if not prompt:
        return ""

    try:
        # 1. 앞뒤 공백 제거
        prompt = prompt.strip()

        # 2. 연속된 공백을 하나로
        import re
        prompt = re.sub(r'\s+', ' ', prompt)

        # 3. 제어 문자 제거 (줄바꿈 제외)
        prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\n\r\t')

        # 4. 유니코드 정규화 (선택적)
        # import unicodedata
        # prompt = unicodedata.normalize('NFKC', prompt)

        return prompt

    except Exception as e:
        logger.warning(f"프롬프트 정제 실패: {e}")
        return prompt.strip() if prompt else ""


def is_prompt_safe(prompt: str) -> tuple[bool, Optional[str]]:
    """
    ✅ 프롬프트 안전성 검사 (선제적 필터링)

    Returns:
        (is_safe, warning_message)
    """
    if not prompt:
        return True, None

    # 위험한 키워드 패턴 (예시)
    dangerous_patterns = [
        "violent", "blood", "gore", "explicit",
        "nude", "nsfw", "sexual"
    ]

    prompt_lower = prompt.lower()

    for pattern in dangerous_patterns:
        if pattern in prompt_lower:
            logger.warning(f"잠재적 위험 키워드 감지: {pattern}")
            return False, f"프롬프트에 부적절한 내용이 포함되어 있을 수 있습니다"

    return True, None


# =========================
# 입력 검증 함수
# =========================
def validate_parameters(
        models: str,
        prompt: str,
        size: str,
        quality: str,
        style: str
) -> tuple[bool, str]:
    """
    ✅ 모든 입력 파라미터 검증

    Returns:
        (is_valid, error_message)
    """
    # 1. 모델 검증
    if not models:
        return False, "모델명이 비어있습니다"

    if not isinstance(models, str):
        return False, f"모델명이 문자열이 아닙니다 (타입: {type(models).__name__})"

    models = models.strip()
    if not models:
        return False, "모델명이 공백으로만 구성되어 있습니다"

    if models not in SUPPORTED_MODELS:
        return False, f"지원하지 않는 모델입니다: {models} (지원: {', '.join(SUPPORTED_MODELS)})"

    # 2. 프롬프트 검증
    if prompt is None:
        return False, "프롬프트가 None입니다"

    if not isinstance(prompt, str):
        return False, f"프롬프트가 문자열이 아닙니다 (타입: {type(prompt).__name__})"

    prompt_cleaned = prompt.strip()
    if not prompt_cleaned:
        return False, "프롬프트가 비어있습니다"

    prompt_len = len(prompt_cleaned)
    if prompt_len < MIN_PROMPT_LENGTH:
        return False, f"프롬프트가 너무 짧습니다 (최소: {MIN_PROMPT_LENGTH}자)"

    if prompt_len > MAX_PROMPT_LENGTH:
        return False, f"프롬프트가 너무 깁니다 (최대: {MAX_PROMPT_LENGTH}자, 현재: {prompt_len}자)"

    # 3. 크기 검증
    if not size:
        return False, "이미지 크기가 비어있습니다"

    if not isinstance(size, str):
        return False, f"이미지 크기가 문자열이 아닙니다 (타입: {type(size).__name__})"

    size = size.strip()
    if models in SUPPORTED_SIZES and size not in SUPPORTED_SIZES[models]:
        return False, f"{models}에서 지원하지 않는 크기입니다: {size} (지원: {', '.join(SUPPORTED_SIZES[models])})"

    # 4. 품질 검증 (dall-e-3만 해당)
    if models == "dall-e-3":
        if not quality or not isinstance(quality, str):
            return False, "품질 설정이 올바르지 않습니다"

        quality = quality.strip()
        if quality not in SUPPORTED_QUALITIES:
            return False, f"지원하지 않는 품질입니다: {quality} (지원: {', '.join(SUPPORTED_QUALITIES)})"

    # 5. 스타일 검증 (dall-e-3만 해당)
    if models == "dall-e-3":
        if not style or not isinstance(style, str):
            return False, "스타일 설정이 올바르지 않습니다"

        style = style.strip()
        if style not in SUPPORTED_STYLES:
            return False, f"지원하지 않는 스타일입니다: {style} (지원: {', '.join(SUPPORTED_STYLES)})"

    return True, ""


def validate_response(resp: Any) -> tuple[bool, str, Optional[str], Optional[str]]:
    """
    ✅ OpenAI 응답 검증 및 파싱

    Returns:
        (is_valid, error_message, image_url, revised_prompt)
    """
    # 1. 응답 존재 확인
    if not resp:
        return False, "OpenAI 응답이 비어있습니다", None, None

    # 2. data 속성 확인
    if not hasattr(resp, 'data'):
        return False, f"OpenAI 응답 구조가 잘못되었습니다 (data 속성 없음, 타입: {type(resp).__name__})", None, None

    # 3. data가 리스트인지 확인
    if not isinstance(resp.data, list):
        return False, f"OpenAI 응답 데이터가 리스트가 아닙니다 (타입: {type(resp.data).__name__})", None, None

    # 4. data가 비어있지 않은지 확인
    if len(resp.data) == 0:
        return False, "OpenAI 응답 데이터가 비어있습니다", None, None

    # 5. 첫 번째 아이템 추출
    try:
        first_item = resp.data[0]
    except (IndexError, TypeError) as e:
        return False, f"응답 데이터 접근 실패: {str(e)}", None, None

    # 6. 이미지 URL 추출
    image_url = None
    try:
        if hasattr(first_item, 'url'):
            image_url = first_item.url
        elif isinstance(first_item, dict):
            image_url = first_item.get('url')

        if not image_url:
            return False, "이미지 URL이 없습니다", None, None

        if not isinstance(image_url, str):
            return False, f"이미지 URL이 문자열이 아닙니다 (타입: {type(image_url).__name__})", None, None

        if not image_url.strip():
            return False, "이미지 URL이 비어있습니다", None, None

        # URL 형식 검증 (기본)
        if not image_url.startswith(('http://', 'https://')):
            logger.warning(f"이미지 URL이 http(s)로 시작하지 않음: {image_url[:50]}")

    except Exception as e:
        return False, f"이미지 URL 추출 실패: {str(e)}", None, None

    # 7. Revised Prompt 추출 (선택적)
    revised_prompt = None
    try:
        if hasattr(first_item, 'revised_prompt'):
            revised_prompt = first_item.revised_prompt
        elif isinstance(first_item, dict):
            revised_prompt = first_item.get('revised_prompt')

        if revised_prompt and isinstance(revised_prompt, str):
            revised_prompt = revised_prompt.strip() if revised_prompt.strip() else None

    except Exception as e:
        logger.debug(f"Revised prompt 추출 실패 (무시 가능): {e}")
        revised_prompt = None

    return True, "", image_url, revised_prompt


# =========================
# 메인 이미지 생성 함수
# =========================
def generate_image(
        models: str = "dall-e-3",
        prompt: str = "",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
) -> Dict[str, Any]:
    """
    ✅ DALL-E 이미지 생성 함수 (모든 예외 케이스 처리)
    ✅ 절대 예외를 밖으로 던지지 않음
    ✅ 재시도 로직 포함

    Args:
        models: DALL-E 모델명
        prompt: 이미지 생성 프롬프트
        size: 이미지 크기
        quality: 이미지 품질 (dall-e-3만 해당)
        style: 이미지 스타일 (dall-e-3만 해당)

    Returns:
        Dict[str, Any]: {
            "image_url": str | None,
            "refined_content": str | None,
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
        error_msg = f"OpenAI 클라이언트가 초기화되지 않았습니다: {client_init_error}"
        logger.error("[VALIDATION] %s", error_msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": error_msg,
        }

    # =========================
    # 2. 입력 검증
    # =========================
    # is_valid, validation_error = validate_parameters(models, prompt, size, quality, style)
    #
    # if not is_valid:
    #     logger.warning("[VALIDATION] %s", validation_error)
    #     return {
    #         "image_url": None,
    #         "refined_content": None,
    #         "error_message": validation_error,
    #     }

    # =========================
    # 3. 프롬프트 정제
    # =========================
    prompt = sanitize_prompt(prompt)

    if not prompt:
        error_msg = "프롬프트 정제 후 비어있습니다"
        logger.warning("[VALIDATION] %s", error_msg)
        return {
            "image_url": None,
            "refined_content": None,
            "error_message": error_msg,
        }

    # =========================
    # 4. 프롬프트 안전성 검사 (선택적)
    # =========================
    is_safe, safety_warning = is_prompt_safe(prompt)
    if not is_safe:
        logger.warning("[SAFETY] %s | prompt=%s...", safety_warning, prompt[:50])
        # 경고만 하고 계속 진행 (OpenAI가 최종 판단)

    # =========================
    # 5. API 파라미터 준비
    # =========================
    api_params = {
        "model": models,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }

    # dall-e-3 전용 파라미터
    if models == "dall-e-3":
        api_params["quality"] = quality
        api_params["style"] = style

    # =========================
    # 6. 재시도 로직을 포함한 API 호출
    # =========================
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                "[API_CALL] 시도 %d/%d | prompt=%s...",
                attempt + 1, MAX_RETRIES, prompt[:100]
            )

            # API 호출
            resp = client.images.generate(**api_params)

            logger.info("[API_CALL] DALL-E API 호출 성공")

            # =========================
            # 7. 응답 검증 및 파싱
            # =========================
            is_valid, error_msg, image_url, revised_prompt = validate_response(resp)

            if not is_valid:
                logger.error("[PARSE] %s", error_msg)
                return {
                    "image_url": None,
                    "refined_content": None,
                    "error_message": error_msg,
                }

            # =========================
            # 8. 성공 응답
            # =========================
            logger.info("[SUCCESS] 이미지 생성 완료 | url=%s...", image_url[:60])
            return {
                "image_url": image_url,
                "refined_content": revised_prompt,
                "error_message": None,
            }

        # =========================
        # 9. 인증 오류 (재시도 불가)
        # =========================
        except AuthenticationError as e:
            error_detail = str(e)
            logger.error("[ERROR] API 인증 실패 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "OpenAI API 키가 유효하지 않습니다. 관리자에게 문의하세요.",
            }

        # =========================
        # 10. 권한 오류 (재시도 불가)
        # =========================
        except PermissionDeniedError as e:
            error_detail = str(e)
            logger.error("[ERROR] API 권한 거부 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "API 사용 권한이 없습니다. 관리자에게 문의하세요.",
            }

        # =========================
        # 11. 잘못된 요청 (재시도 불가)
        # =========================
        except BadRequestError as e:
            error_detail = str(e)

            # 콘텐츠 정책 위반
            if "content_policy_violation" in error_detail.lower():
                logger.warning("[POLICY] 콘텐츠 정책 위반 | prompt=%s...", prompt[:100])
                return {
                    "image_url": None,
                    "refined_content": None,
                    "error_message": "콘텐츠 정책에 위반되어 이미지를 생성할 수 없습니다.",
                }

            # 프롬프트 길이 오류
            if "prompt" in error_detail.lower() and (
                    "too long" in error_detail.lower() or "too short" in error_detail.lower()):
                logger.warning("[ERROR] 프롬프트 길이 오류 | error=%s", error_detail)
                return {
                    "image_url": None,
                    "refined_content": None,
                    "error_message": f"프롬프트 길이가 적절하지 않습니다: {error_detail}",
                }

            # 잘못된 파라미터
            logger.error("[ERROR] BadRequest | params=%s | error=%s", api_params, error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": f"잘못된 요청: {error_detail}",
            }

        # =========================
        # 12. 리소스 없음 (재시도 불가)
        # =========================
        except NotFoundError as e:
            error_detail = str(e)
            logger.error("[ERROR] 리소스를 찾을 수 없음 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "요청한 리소스를 찾을 수 없습니다.",
            }

        # =========================
        # 13. 처리 불가능한 엔티티 (재시도 불가)
        # =========================
        except UnprocessableEntityError as e:
            error_detail = str(e)
            logger.error("[ERROR] 처리 불가능한 요청 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": f"요청을 처리할 수 없습니다: {error_detail}",
            }

        # =========================
        # 14. 충돌 오류 (재시도 불가)
        # =========================
        except ConflictError as e:
            error_detail = str(e)
            logger.error("[ERROR] 충돌 오류 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": f"요청 충돌: {error_detail}",
            }

        # =========================
        # 15. 사용량 초과 (재시도 가능)
        # =========================
        except RateLimitError as e:
            error_detail = str(e)
            last_error = f"API 사용량 초과: {error_detail}"

            logger.warning(
                "[ERROR] API 사용량 초과 (시도 %d/%d) | error=%s",
                attempt + 1, MAX_RETRIES, error_detail
            )

            if attempt < MAX_RETRIES - 1:
                logger.info(f"재시도 대기 중... ({RETRY_DELAY}초)")
                time.sleep(RETRY_DELAY)
                continue

        # =========================
        # 16. 타임아웃 (재시도 가능)
        # =========================
        except APITimeoutError as e:
            error_detail = str(e)
            last_error = f"타임아웃: {error_detail}"

            logger.warning(
                "[ERROR] API 타임아웃 (시도 %d/%d) | error=%s",
                attempt + 1, MAX_RETRIES, error_detail
            )

            if attempt < MAX_RETRIES - 1:
                logger.info(f"재시도 대기 중... ({RETRY_DELAY}초)")
                time.sleep(RETRY_DELAY)
                continue

        # =========================
        # 17. 연결 오류 (재시도 가능)
        # =========================
        except APIConnectionError as e:
            error_detail = str(e)
            last_error = f"연결 실패: {error_detail}"

            logger.warning(
                "[ERROR] API 연결 실패 (시도 %d/%d) | error=%s",
                attempt + 1, MAX_RETRIES, error_detail
            )

            if attempt < MAX_RETRIES - 1:
                logger.info(f"재시도 대기 중... ({RETRY_DELAY}초)")
                time.sleep(RETRY_DELAY)
                continue

        # =========================
        # 18. 서버 내부 오류 (재시도 가능)
        # =========================
        except InternalServerError as e:
            error_detail = str(e)
            last_error = f"서버 오류: {error_detail}"

            logger.warning(
                "[ERROR] OpenAI 서버 오류 (시도 %d/%d) | error=%s",
                attempt + 1, MAX_RETRIES, error_detail
            )

            if attempt < MAX_RETRIES - 1:
                logger.info(f"재시도 대기 중... ({RETRY_DELAY}초)")
                time.sleep(RETRY_DELAY)
                continue

        # =========================
        # 19. 기타 OpenAI 오류
        # =========================
        except APIError as e:
            error_detail = str(e)
            last_error = f"API 오류: {error_detail}"

            logger.error(
                "[ERROR] OpenAI API 오류 (시도 %d/%d) | error=%s",
                attempt + 1, MAX_RETRIES, error_detail
            )

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue

        except OpenAIError as e:
            error_detail = str(e)
            last_error = f"OpenAI 오류: {error_detail}"

            logger.error(
                "[ERROR] OpenAI 오류 (시도 %d/%d) | error=%s",
                attempt + 1, MAX_RETRIES, error_detail
            )

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue

        # =========================
        # 20. JSON 파싱 오류
        # =========================
        except json.JSONDecodeError as e:
            error_detail = str(e)
            logger.error("[ERROR] JSON 파싱 오류 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "응답 파싱 오류가 발생했습니다.",
            }

        # =========================
        # 21. 인코딩 오류
        # =========================
        except UnicodeDecodeError as e:
            error_detail = str(e)
            logger.error("[ERROR] 인코딩 오류 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "응답 인코딩 오류가 발생했습니다.",
            }

        # =========================
        # 22. 메모리 오류
        # =========================
        except MemoryError as e:
            error_detail = str(e)
            logger.critical("[ERROR] 메모리 부족 | error=%s", error_detail)
            return {
                "image_url": None,
                "refined_content": None,
                "error_message": "서버 메모리가 부족합니다.",
            }

        # =========================
        # 23. 예상치 못한 오류
        # =========================
        except Exception as e:
            error_detail = str(e)
            last_error = f"예상치 못한 오류: {error_detail}"

            logger.exception(
                "[ERROR] 예상치 못한 오류 (시도 %d/%d)",
                attempt + 1, MAX_RETRIES
            )

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue

    # =========================
    # 24. 모든 재시도 실패
    # =========================
    logger.error("[ERROR] 모든 재시도 실패 | 마지막 오류=%s", last_error)
    return {
        "image_url": None,
        "refined_content": None,
        "error_message": f"이미지 생성 실패 ({MAX_RETRIES}회 시도): {last_error}",
    }


# =========================
# 헬스체크 함수
# =========================
def health_check() -> dict:
    """OpenAI 서비스 상태 확인"""
    return {
        "client_initialized": client is not None,
        "init_error": client_init_error,
        "api_key_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY.strip()),
        "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
        "max_retries": MAX_RETRIES,
    }