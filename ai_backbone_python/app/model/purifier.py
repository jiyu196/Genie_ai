# app/model/purifier.py
import re
import torch
import logging
import sys
import unicodedata
from typing import Optional

from app.model.model_loader import (
    _tokenizer, _model, _model_loaded, DEVICE, MAX_INPUT_LEN,
    is_model_available
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
DEFAULT_FALLBACK = "이미지 생성 요청"  # 빈 입력 시 기본값

# 지원하는 인코딩 목록
SUPPORTED_ENCODINGS = ['utf-8', 'cp949', 'euc-kr', 'latin-1']


# =========================
# 인코딩 처리 함수
# =========================
def fix_encoding(text: str) -> str:
    """
    ✅ 인코딩 문제 자동 수정
    ✅ 깨진 문자 복구 시도
    """
    if not text:
        return ""

    try:
        # 1. 이미 올바른 UTF-8인지 확인
        text.encode('utf-8').decode('utf-8')
        return text

    except (UnicodeDecodeError, UnicodeEncodeError):
        # 인코딩 문제 발생
        logger.warning("인코딩 문제 감지, 복구 시도 중...")

    # 2. 여러 인코딩으로 복구 시도
    for encoding in SUPPORTED_ENCODINGS:
        try:
            # bytes로 변환 후 다시 디코딩
            if isinstance(text, str):
                # str -> bytes (잘못된 인코딩일 수 있음)
                text_bytes = text.encode('latin-1', errors='ignore')
            else:
                text_bytes = text

            # bytes -> str (올바른 인코딩으로)
            decoded = text_bytes.decode(encoding, errors='ignore')

            if decoded and decoded.strip():
                logger.info(f"인코딩 복구 성공: {encoding}")
                return decoded

        except Exception as e:
            logger.debug(f"{encoding} 인코딩 실패: {e}")
            continue

    # 3. 모든 방법 실패 시 에러 무시하고 변환
    try:
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    except:
        logger.error("인코딩 복구 실패, 원본 반환")
        return text


def normalize_unicode(text: str) -> str:
    """
    ✅ 유니코드 정규화 (NFC, NFD, NFKC, NFKD)
    ✅ 호환 문자를 표준 형태로 변환
    """
    if not text:
        return ""

    try:
        # NFKC: 호환 문자를 표준으로, 결합 (가장 일반적)
        normalized = unicodedata.normalize('NFKC', text)

        # 제어 문자 제거 (C0, C1 제어 코드)
        normalized = ''.join(
            char for char in normalized
            if unicodedata.category(char)[0] != 'C' or char in '\n\r\t '
        )

        return normalized

    except Exception as e:
        logger.warning(f"유니코드 정규화 실패: {e}")
        return text


def remove_invalid_chars(text: str) -> str:
    """
    ✅ 유효하지 않은 문자 제거
    ✅ 출력 불가능한 문자 제거
    """
    if not text:
        return ""

    try:
        # 1. 제어 문자 제거 (줄바꿈, 탭, 공백 제외)
        text = ''.join(
            char for char in text
            if ord(char) >= 32 or char in '\n\r\t'
        )

        # 2. Private Use Area 문자 제거
        text = ''.join(
            char for char in text
            if not ('\uE000' <= char <= '\uF8FF' or
                    '\U000F0000' <= char <= '\U000FFFFD' or
                    '\U00100000' <= char <= '\U0010FFFD')
        )

        # 3. 대체 문자(�) 제거
        text = text.replace('\ufffd', '')

        # 4. 영폭 공백 등 특수 공백 제거
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
    """
    ✅ 더미 토큰 제거
    ✅ None-safe, 인코딩 안전
    """
    if not text:
        return ""

    try:
        # 한글 더미 토큰 제거
        text = re.sub(r"(그그그|으으)", "", text)

        # 영어 더미 토큰 (선택적)
        text = re.sub(r"(aaa|bbb|xxx)", "", text, flags=re.IGNORECASE)

        # 반복되는 공백 정리
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    except Exception as e:
        logger.warning(f"더미 토큰 제거 실패: {e}")
        return text.strip() if text else ""


def keep_before_first_period(text: str) -> str:
    """
    ✅ 첫 마침표 이전까지만 유지
    ✅ None-safe, 빈 문자열 처리
    """
    if not text:
        return ""

    try:
        # 한글 마침표 포함
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


def validate_input_text(text: str) -> tuple[bool, Optional[str]]:
    """
    ✅ 입력 텍스트 검증

    Returns:
        (is_valid, error_message)
    """
    # 1. None 체크
    if text is None:
        return False, "입력 텍스트가 None입니다"

    # 2. 타입 체크
    if not isinstance(text, str):
        return False, f"입력 텍스트가 문자열이 아닙니다 (타입: {type(text).__name__})"

    # 3. 빈 문자열 체크 (공백만 있는 경우 포함)
    if not text.strip():
        return False, "입력 텍스트가 비어있습니다"

    # 4. 길이 체크
    text_len = len(text.strip())
    if text_len < MIN_TEXT_LENGTH:
        return False, f"입력 텍스트가 너무 짧습니다 (최소: {MIN_TEXT_LENGTH}자)"

    if text_len > MAX_TEXT_LENGTH:
        return False, f"입력 텍스트가 너무 깁니다 (최대: {MAX_TEXT_LENGTH}자, 현재: {text_len}자)"

    # 5. 인코딩 확인
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return False, "입력 텍스트에 인코딩 오류가 있습니다"

    return True, None


def sanitize_text(text: str) -> str:
    """
    ✅ 텍스트 정제 (입력 전처리)
    ✅ 인코딩 문제 해결
    ✅ 특수문자 처리
    """
    if not text:
        return ""

    try:
        # 1. 인코딩 수정
        text = fix_encoding(text)

        # 2. 유니코드 정규화
        text = normalize_unicode(text)

        # 3. 유효하지 않은 문자 제거
        text = remove_invalid_chars(text)

        # 4. 앞뒤 공백 제거
        text = text.strip()

        # 5. 연속된 공백을 하나로
        text = re.sub(r'\s+', ' ', text)

        # 6. 연속된 특수문자 정리
        text = re.sub(r'[!?]{3,}', '!!', text)
        text = re.sub(r'\.{3,}', '...', text)

        return text

    except Exception as e:
        logger.warning(f"텍스트 정제 실패: {e}")
        # 최소한의 처리
        try:
            return text.strip()
        except:
            return ""


# =========================
# 모델 추론
# =========================
def purify_sentence(text: str) -> str:
    """
    ✅ 모델을 사용한 텍스트 정제
    ✅ GPU 메모리 관리
    ✅ 인코딩 안전
    ✅ 모든 예외 처리
    """
    # 1. 모델 사용 가능 여부 확인
    if not is_model_available():
        raise RuntimeError("모델이 로드되지 않았습니다")

    # ✅ 전역 변수에서 모델 가져오기
    if _tokenizer is None or _model is None:
        raise RuntimeError("모델 또는 토크나이저가 초기화되지 않았습니다")

    # 2. 입력 정제
    text = sanitize_text(text)

    if not text:
        raise ValueError("정제 후 텍스트가 비어있습니다")

    try:
        # 3. 토크나이징
        logger.debug(f"토크나이징 시작: '{text[:50]}...'")

        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN,
            padding=False
        )

        # token_type_ids 제거 (BART 모델은 사용하지 않음)
        inputs.pop("token_type_ids", None)

        # 4. GPU로 이동
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 5. 추론
        logger.debug("모델 추론 시작")

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_length=64,
                do_sample=False,
                num_beams=1,
                repetition_penalty=2.0,
                no_repeat_ngram_size=4,
                eos_token_id=_tokenizer.eos_token_id,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # 6. 디코딩
        result = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"디코딩 완료: '{result[:50]}...'")

        # 7. GPU 메모리 정리
        if DEVICE == "cuda":
            del inputs
            del outputs
            torch.cuda.empty_cache()

        return result

    except RuntimeError as e:
        # GPU 메모리 부족 등
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        error_msg = f"모델 추론 중 런타임 오류: {str(e)}"
        logger.error(error_msg)

        if "out of memory" in str(e).lower():
            logger.error("💥 GPU 메모리 부족!")

        raise RuntimeError(error_msg) from e

    except UnicodeDecodeError as e:
        # 디코딩 오류
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        error_msg = f"디코딩 오류: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    except Exception as e:
        # 기타 예외
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        error_msg = f"모델 추론 실패: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


# =========================
# 메인 진입점
# =========================
def refine(text: str) -> str:
    """
    ✅ 텍스트 정제 메인 함수
    ✅ is_slang=true일 때 호출
    ✅ 모든 예외를 안전하게 처리
    ✅ 인코딩 문제 완벽 대응
    ✅ 빈 입력 완벽 처리

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트 (실패 시 원본 또는 기본값)
    """
    # 0. None 처리 (최우선)
    if text is None:
        logger.warning("[PURIFIER] 입력이 None | 기본값 반환")
        return DEFAULT_FALLBACK

    # 1. 타입 변환 시도
    if not isinstance(text, str):
        logger.warning(
            "[PURIFIER] 입력이 문자열이 아님 (타입: %s) | 문자열 변환 시도",
            type(text).__name__
        )
        try:
            text = str(text)
        except Exception as e:
            logger.error("[PURIFIER] 문자열 변환 실패: %s | 기본값 반환", e)
            return DEFAULT_FALLBACK

    # 원본 보존
    original_text = text

    # 2. 빈 문자열 처리 (공백만 있는 경우 포함)
    if not text or not text.strip():
        logger.warning("[PURIFIER] 입력이 비어있음 | 기본값 반환")
        return DEFAULT_FALLBACK

    # 3. 인코딩 문제 사전 처리
    try:
        text = fix_encoding(text)
        if not text or not text.strip():
            logger.warning("[PURIFIER] 인코딩 수정 후 비어있음 | 기본값 반환")
            return DEFAULT_FALLBACK
    except Exception as e:
        logger.error("[PURIFIER] 인코딩 수정 실패: %s | 원본 사용", e)

    # 4. 입력 검증
    is_valid, error_msg = validate_input_text(text)

    if not is_valid:
        logger.warning(
            "[PURIFIER] 입력 검증 실패 | error=%s | text='%s'",
            error_msg,
            text[:50] if text else "None"
        )

        # 빈 입력이면 기본값 반환
        if not text or not text.strip():
            return DEFAULT_FALLBACK

        # 길이 초과인 경우 잘라서 사용
        if "너무 깁니다" in error_msg:
            logger.info("[PURIFIER] 입력 길이 초과 | 자르기 수행")
            text = text[:MAX_TEXT_LENGTH].strip()
            if not text:
                return DEFAULT_FALLBACK
        else:
            # 기타 검증 실패 시 원본 반환
            return original_text.strip() if original_text else DEFAULT_FALLBACK

    # 5. 모델 사용 가능 여부 확인
    if not is_model_available():
        logger.warning(
            "[PURIFIER] 모델 미사용 (로드 실패) | text='%s'",
            text[:50]
        )
        return text.strip()

    try:
        # 6. 모델 추론
        logger.info(
            "[PURIFIER] 정제 시작 | text='%s' | len=%d",
            text[:50],
            len(text)
        )

        purified = purify_sentence(text)

        # 7. 후처리
        purified = remove_dummy_tokens(purified)
        purified = keep_before_first_period(purified)

        # 8. 결과 검증
        if not purified or not purified.strip():
            logger.warning(
                "[PURIFIER] 정제 결과가 비어있음 → 원본 반환 | text='%s'",
                text[:50]
            )
            return text.strip() if text.strip() else DEFAULT_FALLBACK

        # 9. 인코딩 재확인 (출력 안전성)
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
                return text.strip() if text.strip() else DEFAULT_FALLBACK

        # 10. 로그 출력
        logger.info(
            "[PURIFIER] 정제 완료 | before='%s' | after='%s' | changed=%s",
            original_text[:50],
            purified[:50],
            purified != original_text
        )

        return purified.strip()

    except ValueError as e:
        # 입력 오류
        logger.warning(
            "[PURIFIER] 입력 오류 → 기본값 반환 | error=%s",
            str(e)
        )
        return DEFAULT_FALLBACK

    except RuntimeError as e:
        # 모델 추론 오류
        logger.error(
            "[PURIFIER] 추론 오류 → 원본 반환 | text='%s' | error=%s",
            text[:50],
            str(e)
        )
        return text.strip() if text.strip() else DEFAULT_FALLBACK

    except UnicodeDecodeError as e:
        # 인코딩/디코딩 오류
        logger.error(
            "[PURIFIER] 인코딩 오류 → 원본 반환 | error=%s",
            str(e)
        )
        return text.strip() if text.strip() else DEFAULT_FALLBACK

    except MemoryError as e:
        # 메모리 부족
        logger.critical(
            "[PURIFIER] 메모리 부족 → 원본 반환 | error=%s",
            str(e)
        )
        # GPU 메모리 정리
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return text.strip() if text.strip() else DEFAULT_FALLBACK

    except Exception as e:
        # 예상치 못한 오류
        logger.exception(
            "[PURIFIER] 예상치 못한 오류 → 원본 반환 | text='%s'",
            text[:50] if text else "None"
        )
        return text.strip() if text and text.strip() else DEFAULT_FALLBACK


# =========================
# 헬스체크 함수
# =========================
def health_check() -> dict:
    """
    ✅ Purifier 상태 확인
    """
    return {
        "model_available": is_model_available(),
        "model_loaded": _model_loaded,
        "device": DEVICE,
        "max_input_len": MAX_INPUT_LEN,
        "min_text_length": MIN_TEXT_LENGTH,
        "max_text_length": MAX_TEXT_LENGTH,
        "default_fallback": DEFAULT_FALLBACK,
    }