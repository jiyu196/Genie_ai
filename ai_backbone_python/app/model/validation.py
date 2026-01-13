# app/model/validation.py
"""
텍스트 정제 결과 검증 모듈
- 과도한 변형 감지
- 의미 손실 방지
- 적응형 임계값
"""
import re
import logging
from typing import Tuple, Optional
from difflib import SequenceMatcher

logger = logging.getLogger("validation")


# =========================
# 설정값
# =========================
class ValidationConfig:
    """검증 설정 (텍스트 길이에 따라 적응)"""

    # 길이별 임계값 (original_length -> (min_ratio, max_ratio))
    LENGTH_THRESHOLDS = {
        10: (0.5, 2.2),    # 매우 짧은 텍스트: 관대한 기준
        20: (0.5, 2.0),    # 짧은 텍스트
        50: (0.5, 2.0),    # 중간 텍스트
        100: (0.6, 1.8),   # 긴 텍스트
        200: (0.7, 1.6),   # 매우 긴 텍스트: 엄격한 기준
    }

    # 유사도 임계값
    MIN_SIMILARITY = 0.3  # 최소 30% 유사해야 함

    # 의미 있는 단어 최소 보존율
    MIN_MEANINGFUL_WORD_RETENTION = 0.4  # 40% 이상의 의미 있는 단어 보존

    # 특수 케이스
    MAX_ABSOLUTE_LENGTH = 500  # 절대 최대 길이
    MIN_ABSOLUTE_LENGTH = 1    # 절대 최소 길이


# =========================
# 유틸리티 함수
# =========================
def get_adaptive_thresholds(text_length: int) -> Tuple[float, float]:
    """
    텍스트 길이에 따른 적응형 임계값 계산

    Args:
        text_length: 원본 텍스트 길이

    Returns:
        (min_ratio, max_ratio) 튜플
    """
    thresholds = ValidationConfig.LENGTH_THRESHOLDS

    # 정렬된 길이 기준점
    length_points = sorted(thresholds.keys())

    # 가장 짧은 것보다 짧으면
    if text_length <= length_points[0]:
        return thresholds[length_points[0]]

    # 가장 긴 것보다 길면
    if text_length >= length_points[-1]:
        return thresholds[length_points[-1]]

    # 사이값은 선형 보간
    for i in range(len(length_points) - 1):
        lower = length_points[i]
        upper = length_points[i + 1]

        if lower <= text_length <= upper:
            # 선형 보간
            lower_min, lower_max = thresholds[lower]
            upper_min, upper_max = thresholds[upper]

            ratio = (text_length - lower) / (upper - lower)

            min_threshold = lower_min + (upper_min - lower_min) * ratio
            max_threshold = lower_max + (upper_max - lower_max) * ratio

            return min_threshold, max_threshold

    # 기본값 (도달하지 않아야 함)
    return 0.5, 2.0


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트의 유사도 계산 (0.0 ~ 1.0)
    SequenceMatcher 사용
    """
    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(None, text1, text2).ratio()


def extract_meaningful_words(text: str) -> set:
    """
    의미 있는 단어 추출 (명사, 동사 위주)
    - 한글: 2자 이상
    - 영어: 3자 이상
    - 숫자 제외
    """
    if not text:
        return set()

    # 공백으로 분리
    words = text.split()

    meaningful = set()
    for word in words:
        # 정제: 특수문자 제거
        cleaned = re.sub(r'[^\w\s]', '', word)

        if not cleaned:
            continue

        # 한글 단어 (2자 이상)
        if re.search(r'[가-힣]{2,}', cleaned):
            meaningful.add(cleaned)

        # 영어 단어 (3자 이상, 숫자 제외)
        elif re.search(r'[a-zA-Z]{3,}', cleaned) and not cleaned.isdigit():
            meaningful.add(cleaned.lower())

    return meaningful


def calculate_word_retention(original: str, purified: str) -> float:
    """
    의미 있는 단어 보존율 계산

    Returns:
        0.0 ~ 1.0 (1.0 = 모든 단어 보존)
    """
    original_words = extract_meaningful_words(original)
    purified_words = extract_meaningful_words(purified)

    if not original_words:
        return 1.0  # 원본에 의미 있는 단어가 없으면 통과

    # 보존된 단어 수
    retained = len(original_words & purified_words)

    return retained / len(original_words)


def has_critical_information_loss(original: str, purified: str) -> bool:
    """
    치명적인 정보 손실 여부 확인
    - 핵심 키워드 누락
    - 숫자/날짜 변경
    - 고유명사 누락 등
    """
    # 1. 숫자 확인 (숫자가 완전히 사라지거나 변경됨)
    original_numbers = set(re.findall(r'\d+', original))
    purified_numbers = set(re.findall(r'\d+', purified))

    if original_numbers and not purified_numbers:
        logger.debug("[CRITICAL_LOSS] 모든 숫자 정보 손실")
        return True

    if original_numbers and original_numbers != purified_numbers:
        # 숫자가 변경됨 (단, 순서만 바뀐 경우는 허용)
        if len(original_numbers - purified_numbers) > len(original_numbers) * 0.5:
            logger.debug(
                "[CRITICAL_LOSS] 주요 숫자 변경 | original=%s | purified=%s",
                original_numbers,
                purified_numbers
            )
            return True

    # 2. 영어 고유명사 확인 (대문자로 시작하는 단어)
    original_proper = set(re.findall(r'\b[A-Z][a-z]+\b', original))
    purified_proper = set(re.findall(r'\b[A-Z][a-z]+\b', purified))

    if len(original_proper) > 0:
        retention_rate = len(original_proper & purified_proper) / len(original_proper)
        if retention_rate < 0.5:
            logger.debug(
                "[CRITICAL_LOSS] 고유명사 대부분 손실 | retention=%.2f",
                retention_rate
            )
            return True

    # 3. 특수 패턴 확인 (URL, 이메일 등)
    special_patterns = [
        (r'https?://[^\s]+', 'URL'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '이메일'),
        (r'#\w+', '해시태그'),
    ]

    for pattern, name in special_patterns:
        original_matches = set(re.findall(pattern, original))
        purified_matches = set(re.findall(pattern, purified))

        if original_matches and not purified_matches:
            logger.debug(f"[CRITICAL_LOSS] {name} 정보 손실")
            return True

    return False


# =========================
# 메인 검증 함수
# =========================
def is_over_modified(original: str, purified: str) -> Tuple[bool, Optional[str]]:
    """
    텍스트가 과도하게 변형되었는지 종합 검증

    Args:
        original: 원본 텍스트
        purified: 정제된 텍스트

    Returns:
        (is_invalid, reason) 튜플
        - is_invalid: True면 과도한 변형
        - reason: 거부 사유
    """
    if not original or not purified:
        return False, None

    original = original.strip()
    purified = purified.strip()

    original_len = len(original)
    purified_len = len(purified)

    # 1. 절대 길이 체크
    if purified_len > ValidationConfig.MAX_ABSOLUTE_LENGTH:
        reason = f"정제된 텍스트가 너무 김 ({purified_len}자 > {ValidationConfig.MAX_ABSOLUTE_LENGTH}자)"
        logger.debug(f"[OVER_MODIFIED] {reason}")
        return True, reason

    if purified_len < ValidationConfig.MIN_ABSOLUTE_LENGTH:
        reason = f"정제된 텍스트가 너무 짧음 ({purified_len}자)"
        logger.debug(f"[OVER_MODIFIED] {reason}")
        return True, reason

    # 2. 적응형 길이 비율 체크
    len_ratio = purified_len / original_len if original_len > 0 else 0
    min_ratio, max_ratio = get_adaptive_thresholds(original_len)

    if len_ratio < min_ratio or len_ratio > max_ratio:
        reason = (
            f"길이 비율 이상 (ratio={len_ratio:.2f}, "
            f"허용범위=[{min_ratio:.2f}, {max_ratio:.2f}], "
            f"원본={original_len}자, 정제={purified_len}자)"
        )
        logger.debug(f"[OVER_MODIFIED] {reason}")
        return True, reason

    # 3. 텍스트 유사도 체크
    similarity = calculate_text_similarity(original, purified)
    if similarity < ValidationConfig.MIN_SIMILARITY:
        reason = f"유사도 너무 낮음 (similarity={similarity:.2f} < {ValidationConfig.MIN_SIMILARITY})"
        logger.debug(f"[OVER_MODIFIED] {reason}")
        return True, reason

    # 4. 의미 있는 단어 보존율 체크
    word_retention = calculate_word_retention(original, purified)
    if word_retention < ValidationConfig.MIN_MEANINGFUL_WORD_RETENTION:
        reason = (
            f"핵심 단어 보존율 낮음 "
            f"(retention={word_retention:.2f} < {ValidationConfig.MIN_MEANINGFUL_WORD_RETENTION})"
        )
        logger.debug(f"[OVER_MODIFIED] {reason}")
        return True, reason

    # 5. 치명적인 정보 손실 체크
    if has_critical_information_loss(original, purified):
        reason = "치명적인 정보 손실 감지 (숫자, 고유명사, URL 등)"
        logger.debug(f"[OVER_MODIFIED] {reason}")
        return True, reason

    # 통과
    logger.debug(
        "[VALIDATION_PASS] "
        f"len_ratio={len_ratio:.2f}, "
        f"similarity={similarity:.2f}, "
        f"word_retention={word_retention:.2f}"
    )
    return False, None


# =========================
# 검증 통계
# =========================
_validation_stats = {
    "total_checks": 0,
    "rejected_by_length": 0,
    "rejected_by_similarity": 0,
    "rejected_by_word_retention": 0,
    "rejected_by_critical_loss": 0,
    "passed": 0,
}


def record_validation_result(is_rejected: bool, reason: Optional[str] = None):
    """검증 결과 기록"""
    _validation_stats["total_checks"] += 1

    if not is_rejected:
        _validation_stats["passed"] += 1
        return

    if reason:
        if "길이" in reason:
            _validation_stats["rejected_by_length"] += 1
        elif "유사도" in reason:
            _validation_stats["rejected_by_similarity"] += 1
        elif "단어" in reason or "보존" in reason:
            _validation_stats["rejected_by_word_retention"] += 1
        elif "손실" in reason:
            _validation_stats["rejected_by_critical_loss"] += 1


def get_validation_stats() -> dict:
    """검증 통계 반환"""
    total = _validation_stats["total_checks"]
    if total == 0:
        return {**_validation_stats, "rejection_rate": 0.0}

    return {
        **_validation_stats,
        "rejection_rate": (total - _validation_stats["passed"]) / total * 100,
    }


def reset_validation_stats():
    """통계 초기화"""
    for key in _validation_stats:
        _validation_stats[key] = 0


# =========================
# 설정 조정 헬퍼
# =========================
def adjust_thresholds(
        min_similarity: Optional[float] = None,
        min_word_retention: Optional[float] = None,
        length_thresholds: Optional[dict] = None
):
    """
    런타임에 임계값 조정

    Example:
        # 더 관대하게
        adjust_thresholds(min_similarity=0.2, min_word_retention=0.3)

        # 더 엄격하게
        adjust_thresholds(min_similarity=0.5, min_word_retention=0.6)
    """
    if min_similarity is not None:
        ValidationConfig.MIN_SIMILARITY = min_similarity
        logger.info(f"MIN_SIMILARITY 조정: {min_similarity}")

    if min_word_retention is not None:
        ValidationConfig.MIN_MEANINGFUL_WORD_RETENTION = min_word_retention
        logger.info(f"MIN_MEANINGFUL_WORD_RETENTION 조정: {min_word_retention}")

    if length_thresholds is not None:
        ValidationConfig.LENGTH_THRESHOLDS = length_thresholds
        logger.info(f"LENGTH_THRESHOLDS 조정: {length_thresholds}")