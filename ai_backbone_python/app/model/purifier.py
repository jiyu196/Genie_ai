# app/model/purifier.py
import re
import torch
import logging
import sys
import unicodedata
import time
from typing import Optional, Tuple, Union, Dict
from functools import wraps

from app.model.model_loader import (
    get_tokenizer, get_model, is_model_available,
    DEVICE, MAX_INPUT_LEN, gpu_memory_manager
)
from app.model.validation import (
    is_over_modified,
    record_validation_result,
    get_validation_stats as get_val_stats,
    adjust_thresholds
)

# 로거 설정
logger = logging.getLogger("purifier")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

# =========================
# 상수 정의
# =========================
MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 1000
DEFAULT_FALLBACK = "이미지 생성 요청"

# 지원하는 인코딩 목록
SUPPORTED_ENCODINGS = ['utf-8', 'cp949', 'euc-kr', 'latin-1']

# 성능 모니터링
_stats = {
    "total_requests": 0,
    "success_count": 0,
    "error_count": 0,
    "total_time": 0.0,
}


# =========================
# 데코레이터
# =========================
def performance_monitor(func):
    """성능 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _stats["total_requests"] += 1
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            _stats["success_count"] += 1
            return result
        except Exception as e:
            _stats["error_count"] += 1
            raise
        finally:
            elapsed = time.time() - start_time
            _stats["total_time"] += elapsed
            logger.debug(f"{func.__name__} 실행 시간: {elapsed:.3f}초")

    return wrapper


# =========================
# 인코딩 처리 함수
# =========================
def fix_encoding(text: str) -> str:
    """인코딩 문제 자동 수정"""
    if not text:
        return ""

    try:
        # 이미 올바른 UTF-8인지 확인
        text.encode('utf-8').decode('utf-8')
        return text
    except (UnicodeDecodeError, UnicodeEncodeError):
        logger.warning("인코딩 문제 감지, 복구 시도 중...")

    # 여러 인코딩으로 복구 시도
    for encoding in SUPPORTED_ENCODINGS:
        try:
            if isinstance(text, str):
                text_bytes = text.encode('latin-1', errors='ignore')
            else:
                text_bytes = text

            decoded = text_bytes.decode(encoding, errors='ignore')

            if decoded and decoded.strip():
                logger.info(f"인코딩 복구 성공: {encoding}")
                return decoded

        except Exception as e:
            logger.debug(f"{encoding} 인코딩 실패: {e}")
            continue

    # 모든 방법 실패 시
    try:
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    except:
        logger.error("인코딩 복구 실패, 원본 반환")
        return text


def normalize_unicode(text: str) -> str:
    """유니코드 정규화 (NFC, NFD, NFKC, NFKD)"""
    if not text:
        return ""

    try:
        # NFKC: 호환 문자를 표준으로, 결합
        normalized = unicodedata.normalize('NFKC', text)

        # 제어 문자 제거 (줄바꿈, 탭, 공백 제외)
        normalized = ''.join(
            char for char in normalized
            if unicodedata.category(char)[0] != 'C' or char in '\n\r\t '
        )

        return normalized

    except Exception as e:
        logger.warning(f"유니코드 정규화 실패: {e}")
        return text


def remove_invalid_chars(text: str) -> str:
    """유효하지 않은 문자 제거"""
    if not text:
        return ""

    try:
        # 제어 문자 제거
        text = ''.join(
            char for char in text
            if ord(char) >= 32 or char in '\n\r\t'
        )

        # Private Use Area 문자 제거
        text = ''.join(
            char for char in text
            if not ('\uE000' <= char <= '\uF8FF' or
                    '\U000F0000' <= char <= '\U000FFFFD' or
                    '\U00100000' <= char <= '\U0010FFFD')
        )

        # 대체 문자(�) 및 특수 공백 제거
        text = text.replace('\ufffd', '')
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        text = text.replace('\ufeff', '')  # BOM

        return text

    except Exception as e:
        logger.warning(f"유효하지 않은 문자 제거 실패: {e}")
        return text


# =========================
# 유틸리티 함수
# =========================
def remove_dummy_tokens(text: str) -> str:
    """더미 토큰 제거"""
    if not text:
        return ""

    try:
        # 한글 더미 토큰
        text = re.sub(r"(그그그|으으)", "", text)
        # 영어 더미 토큰
        text = re.sub(r"(aaa|bbb|xxx)", "", text, flags=re.IGNORECASE)
        # 공백 정리
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    except Exception as e:
        logger.warning(f"더미 토큰 제거 실패: {e}")
        return text.strip() if text else ""


def keep_before_first_period(text: str) -> str:
    """첫 마침표 이전까지만 유지"""
    if not text:
        return ""

    try:
        periods = ['.', '。', '．']
        min_pos = len(text)

        for period in periods:
            pos = text.find(period)
            if pos != -1 and pos < min_pos:
                min_pos = pos

        if min_pos < len(text):
            text = text[:min_pos]

        return text.strip()

    except Exception as e:
        logger.warning(f"마침표 처리 실패: {e}")
        return text.strip() if text else ""


def validate_input_text(text: str) -> Tuple[bool, Optional[str]]:
    """입력 텍스트 검증"""
    if text is None:
        return False, "입력 텍스트가 None입니다"

    if not isinstance(text, str):
        return False, f"입력 텍스트가 문자열이 아닙니다 (타입: {type(text).__name__})"

    if not text.strip():
        return False, "입력 텍스트가 비어있습니다"

    text_len = len(text.strip())
    if text_len < MIN_TEXT_LENGTH:
        return False, f"입력 텍스트가 너무 짧습니다 (최소: {MIN_TEXT_LENGTH}자)"

    if text_len > MAX_TEXT_LENGTH:
        return False, f"입력 텍스트가 너무 깁니다 (최대: {MAX_TEXT_LENGTH}자, 현재: {text_len}자)"

    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return False, "입력 텍스트에 인코딩 오류가 있습니다"

    return True, None


def sanitize_text(text: str) -> str:
    """텍스트 정제 (전처리)"""
    if not text:
        return ""

    try:
        text = fix_encoding(text)
        text = normalize_unicode(text)
        text = remove_invalid_chars(text)
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[!?]{3,}', '!!', text)
        text = re.sub(r'\.{3,}', '...', text)

        return text

    except Exception as e:
        logger.warning(f"텍스트 정제 실패: {e}")
        try:
            return text.strip()
        except:
            return ""


# =========================
# 모델 추론
# =========================
@performance_monitor
def purify_sentence(text: str) -> str:
    """모델을 사용한 텍스트 정제"""
    if not is_model_available():
        raise RuntimeError("모델이 로드되지 않았습니다")

    tokenizer = get_tokenizer()
    model = get_model()

    if tokenizer is None or model is None:
        raise RuntimeError("모델 또는 토크나이저가 초기화되지 않았습니다")

    # 입력 정제
    text = sanitize_text(text)
    if not text:
        raise ValueError("정제 후 텍스트가 비어있습니다")

    try:
        # 토크나이징
        logger.debug(f"토크나이징: '{text[:50]}...'")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN,
            padding=False
        )

        # token_type_ids 제거
        inputs.pop("token_type_ids", None)

        # GPU로 이동
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 추론
        logger.debug("모델 추론 시작")

        with torch.no_grad():
            with gpu_memory_manager():
                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=4,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

        # 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"디코딩 완료: '{result[:50]}...'")

        # 메모리 정리
        del inputs, outputs
        if "cuda" in DEVICE:
            torch.cuda.empty_cache()

        return result

    except RuntimeError as e:
        if "cuda" in DEVICE:
            torch.cuda.empty_cache()

        error_msg = f"모델 추론 중 런타임 오류: {str(e)}"
        logger.error(error_msg)

        if "out of memory" in str(e).lower():
            logger.error("💥 GPU 메모리 부족!")

        raise RuntimeError(error_msg) from e

    except Exception as e:
        if "cuda" in DEVICE:
            torch.cuda.empty_cache()

        error_msg = f"모델 추론 실패: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# =========================
# 메인 진입점
# =========================
@performance_monitor
def refine(
        text: str,
        skip_validation: bool = False,
        return_metadata: bool = False
) -> Union[str, Dict]:
    """
    텍스트 정제 메인 함수

    Args:
        text: 원본 텍스트
        skip_validation: True면 검증 스킵 (개발/테스트용)
        return_metadata: True면 메타데이터 포함 dict 반환

    Returns:
        정제된 텍스트 (또는 메타데이터 dict)
    """
    # None 처리
    if text is None:
        logger.warning("[PURIFIER] 입력이 None | 기본값 반환")
        if return_metadata:
            return {
                "original": None,
                "purified": DEFAULT_FALLBACK,
                "changed": False,
                "validation_passed": False,
                "validation_reason": "input_is_none",
            }
        return DEFAULT_FALLBACK

    # 타입 변환
    if not isinstance(text, str):
        logger.warning(
            "[PURIFIER] 입력이 문자열이 아님 (타입: %s) | 문자열 변환 시도",
            type(text).__name__
        )
        try:
            text = str(text)
        except Exception as e:
            logger.error("[PURIFIER] 문자열 변환 실패: %s | 기본값 반환", e)
            if return_metadata:
                return {
                    "original": text,
                    "purified": DEFAULT_FALLBACK,
                    "changed": False,
                    "validation_passed": False,
                    "validation_reason": f"type_conversion_failed: {str(e)}",
                }
            return DEFAULT_FALLBACK

    original_text = text

    # 빈 문자열 처리
    if not text or not text.strip():
        logger.warning("[PURIFIER] 입력이 비어있음 | 기본값 반환")
        if return_metadata:
            return {
                "original": original_text,
                "purified": DEFAULT_FALLBACK,
                "changed": False,
                "validation_passed": False,
                "validation_reason": "input_is_empty",
            }
        return DEFAULT_FALLBACK

    # 인코딩 문제 사전 처리
    try:
        text = fix_encoding(text)
        if not text or not text.strip():
            logger.warning("[PURIFIER] 인코딩 수정 후 비어있음 | 기본값 반환")
            if return_metadata:
                return {
                    "original": original_text,
                    "purified": DEFAULT_FALLBACK,
                    "changed": False,
                    "validation_passed": False,
                    "validation_reason": "encoding_fix_resulted_empty",
                }
            return DEFAULT_FALLBACK
    except Exception as e:
        logger.error("[PURIFIER] 인코딩 수정 실패: %s | 원본 사용", e)

    # 입력 검증
    is_valid, error_msg = validate_input_text(text)

    if not is_valid:
        logger.warning(
            "[PURIFIER] 입력 검증 실패 | error=%s | text='%s'",
            error_msg,
            text[:50] if text else "None"
        )

        if not text or not text.strip():
            if return_metadata:
                return {
                    "original": original_text,
                    "purified": DEFAULT_FALLBACK,
                    "changed": False,
                    "validation_passed": False,
                    "validation_reason": error_msg,
                }
            return DEFAULT_FALLBACK

        # 길이 초과 시 자르기
        if "너무 깁니다" in error_msg:
            logger.info("[PURIFIER] 입력 길이 초과 | 자르기 수행")
            text = text[:MAX_TEXT_LENGTH].strip()
            if not text:
                if return_metadata:
                    return {
                        "original": original_text,
                        "purified": DEFAULT_FALLBACK,
                        "changed": False,
                        "validation_passed": False,
                        "validation_reason": "truncated_but_empty",
                    }
                return DEFAULT_FALLBACK
        else:
            result = original_text.strip() if original_text else DEFAULT_FALLBACK
            if return_metadata:
                return {
                    "original": original_text,
                    "purified": result,
                    "changed": False,
                    "validation_passed": False,
                    "validation_reason": error_msg,
                }
            return result

    # 모델 사용 가능 여부 확인
    if not is_model_available():
        logger.warning(
            "[PURIFIER] 모델 미사용 (로드 실패) | text='%s'",
            text[:50]
        )
        result = text.strip()
        if return_metadata:
            return {
                "original": original_text,
                "purified": result,
                "changed": False,
                "validation_passed": False,
                "validation_reason": "model_not_available",
            }
        return result

    try:
        # 모델 추론
        logger.info(
            "[PURIFIER] 정제 시작 | text='%s' | len=%d",
            text[:50],
            len(text)
        )

        purified = purify_sentence(text)

        # 후처리
        purified = remove_dummy_tokens(purified)
        purified = keep_before_first_period(purified)

        # 결과 검증
        if not purified or not purified.strip():
            logger.warning(
                "[PURIFIER] 정제 결과가 비어있음 → 원본 반환 | text='%s'",
                text[:50]
            )
            result = text.strip() if text.strip() else DEFAULT_FALLBACK
            if return_metadata:
                return {
                    "original": original_text,
                    "purified": result,
                    "changed": False,
                    "validation_passed": False,
                    "validation_reason": "purified_result_empty",
                }
            return result

        # ✅ 검증 로직 통합
        if not skip_validation:
            is_invalid, reason = is_over_modified(original_text, purified)
            record_validation_result(is_invalid, reason)

            if is_invalid:
                logger.warning(
                    "[PURIFIER] 검증 실패 → 원본 반환 | reason=%s | "
                    "original='%s' | purified='%s'",
                    reason,
                    original_text[:50],
                    purified[:50]
                )

                if return_metadata:
                    return {
                        "original": original_text,
                        "purified": original_text.strip(),
                        "changed": False,
                        "validation_passed": False,
                        "validation_reason": reason,
                        "attempted_purified": purified,
                    }

                return original_text.strip() if original_text.strip() else DEFAULT_FALLBACK

        # 인코딩 재확인
        try:
            purified.encode('utf-8')
        except UnicodeEncodeError as e:
            logger.warning(
                "[PURIFIER] 결과 인코딩 오류 → 수정 시도 | error=%s",
                str(e)
            )
            purified = fix_encoding(purified)

            if not purified or not purified.strip():
                logger.warning("[PURIFIER] 인코딩 수정 후 비어있음 → 원본 반환")
                result = text.strip() if text.strip() else DEFAULT_FALLBACK
                if return_metadata:
                    return {
                        "original": original_text,
                        "purified": result,
                        "changed": False,
                        "validation_passed": False,
                        "validation_reason": "encoding_fix_after_purify_empty",
                    }
                return result

        # 로그 출력
        logger.info(
            "[PURIFIER] 정제 완료 | before='%s' | after='%s' | changed=%s | validated=%s",
            original_text[:50],
            purified[:50],
            purified != original_text,
            not skip_validation
        )

        if return_metadata:
            return {
                "original": original_text,
                "purified": purified.strip(),
                "changed": purified.strip() != original_text.strip(),
                "validation_passed": True,
                "validation_reason": None,
                "original_length": len(original_text),
                "purified_length": len(purified),
                "length_ratio": len(purified) / len(original_text) if original_text else 0,
            }

        return purified.strip()

    except ValueError as e:
        logger.warning("[PURIFIER] 입력 오류 → 기본값 반환 | error=%s", str(e))
        if return_metadata:
            return {
                "original": original_text,
                "purified": DEFAULT_FALLBACK,
                "changed": False,
                "validation_passed": False,
                "validation_reason": f"value_error: {str(e)}",
            }
        return DEFAULT_FALLBACK

    except RuntimeError as e:
        logger.error(
            "[PURIFIER] 추론 오류 → 원본 반환 | text='%s' | error=%s",
            text[:50],
            str(e)
        )
        result = text.strip() if text.strip() else DEFAULT_FALLBACK
        if return_metadata:
            return {
                "original": original_text,
                "purified": result,
                "changed": False,
                "validation_passed": False,
                "validation_reason": f"runtime_error: {str(e)}",
            }
        return result

    except MemoryError as e:
        logger.critical("[PURIFIER] 메모리 부족 → 원본 반환 | error=%s", str(e))
        if "cuda" in DEVICE:
            torch.cuda.empty_cache()
        result = text.strip() if text.strip() else DEFAULT_FALLBACK
        if return_metadata:
            return {
                "original": original_text,
                "purified": result,
                "changed": False,
                "validation_passed": False,
                "validation_reason": f"memory_error: {str(e)}",
            }
        return result

    except Exception as e:
        logger.exception(
            "[PURIFIER] 예상치 못한 오류 → 원본 반환 | text='%s'",
            text[:50] if text else "None"
        )
        result = text.strip() if text and text.strip() else DEFAULT_FALLBACK
        if return_metadata:
            return {
                "original": original_text,
                "purified": result,
                "changed": False,
                "validation_passed": False,
                "validation_reason": f"unexpected_error: {str(e)}",
            }
        return result


# =========================
# 배치 처리
# =========================
def refine_batch(
        texts: list,
        skip_validation: bool = False,
        return_metadata: bool = False
) -> list:
    """
    여러 텍스트를 배치로 처리

    Args:
        texts: 텍스트 리스트
        skip_validation: 검증 스킵 여부
        return_metadata: 메타데이터 포함 여부

    Returns:
        정제된 텍스트 리스트 (또는 메타데이터 dict 리스트)
    """
    if not texts:
        return []

    results = []
    for text in texts:
        try:
            result = refine(
                text,
                skip_validation=skip_validation,
                return_metadata=return_metadata
            )
            results.append(result)
        except Exception as e:
            logger.error(f"배치 처리 중 오류 | text='{text[:30] if text else None}' | error={e}")
            if return_metadata:
                results.append({
                    "original": text,
                    "purified": text if text else DEFAULT_FALLBACK,
                    "changed": False,
                    "validation_passed": False,
                    "validation_reason": f"batch_error: {str(e)}",
                })
            else:
                results.append(text if text else DEFAULT_FALLBACK)

    return results


# =========================
# 헬스체크 및 통계
# =========================
def health_check() -> dict:
    """Purifier 상태 확인 (검증 통계 포함)"""
    return {
        "model_available": is_model_available(),
        "device": DEVICE,
        "max_input_len": MAX_INPUT_LEN,
        "min_text_length": MIN_TEXT_LENGTH,
        "max_text_length": MAX_TEXT_LENGTH,
        "default_fallback": DEFAULT_FALLBACK,
        "purifier_stats": get_stats(),
        "validation_stats": get_val_stats(),
    }


def get_stats() -> dict:
    """성능 통계 반환"""
    total = _stats["total_requests"]
    if total == 0:
        return {**_stats, "avg_time": 0.0, "success_rate": 0.0}

    return {
        **_stats,
        "avg_time": _stats["total_time"] / total,
        "success_rate": _stats["success_count"] / total * 100,
    }


def reset_stats():
    """통계 초기화"""
    _stats["total_requests"] = 0
    _stats["success_count"] = 0
    _stats["error_count"] = 0
    _stats["total_time"] = 0.0