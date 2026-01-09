"""
후처리 모듈: KoBART 모델 출력 검증 및 보정
"""
import re
import logging
from typing import Optional

logger = logging.getLogger("post_processor")


def post_process(original: str, purified: str) -> str:
    """
    모델 출력 후처리 파이프라인

    Args:
        original: 원본 문장
        purified: 모델이 순화한 문장

    Returns:
        검증 및 보정된 최종 문장
    """
    logger.info(
        "[POST_PROCESS] 시작 | original=%s... | purified=%s...",
        original[:30],
        purified[:30]
    )

    # Step 1: 출력 검증
    if not validate_output(purified):
        logger.warning("[POST_PROCESS] 비정상 출력 감지, 원문 반환 | purified=%s", purified)
        return original

    # Step 2: 과도한 변형 체크
    if is_over_modified(original, purified):
        logger.warning("[POST_PROCESS] 과도한 변형 감지, 원문 반환 | original=%s | purified=%s", original, purified)
        return original

    # Step 3: 의미 보존 체크 (단어 기반)
    if not check_meaning_preservation(original, purified):
        logger.warning("[POST_PROCESS] 의미 손실 감지, 원문 반환 | original=%s | purified=%s", original, purified)
        return original

    # Step 4: 화이트리스트 체크
    if is_whitelisted(original):
        logger.info("[POST_PROCESS] 화이트리스트 문장, 원문 유지 | original=%s", original)
        return original

    # Step 5: 최종 정리
    result = purified.strip()

    logger.info(
        "[POST_PROCESS] 완료 | result=%s... | changed=%s",
        result[:30],
        original != result
    )

    return result


def validate_output(purified: str) -> bool:
    """
    비정상 출력 필터링
    """
    # 체크 1: 빈 문자열
    if not purified or not purified.strip():
        logger.debug("[VALIDATE] 빈 문자열")
        return False

    # 체크 2: 너무 짧음 (2자 미만)
    if len(purified.strip()) < 2:
        logger.debug("[VALIDATE] 너무 짧음: %d자", len(purified.strip()))
        return False

    # 체크 3: 반복 문자 (모델 오작동)
    if re.search(r'(.)\1{4,}', purified):
        logger.debug("[VALIDATE] 반복 문자 감지: %s", purified)
        return False

    # 체크 4: 특수문자만 있음
    if not re.search(r'[가-힣a-zA-Z]', purified):
        logger.debug("[VALIDATE] 특수문자만 존재: %s", purified)
        return False

    # 체크 5: 의미없는 토큰
    invalid_tokens = ['<unk>', '[PAD]', '[UNK]', '<pad>', '<s>', '</s>']
    if any(token in purified for token in invalid_tokens):
        logger.debug("[VALIDATE] 의미없는 토큰 발견: %s", purified)
        return False

    return True


def is_over_modified(original: str, purified: str) -> bool:
    """
    과도한 변형 감지
    """
    original_len = len(original.strip())
    purified_len = len(purified.strip())

    # 길이 비율 체크
    if original_len == 0:
        return False

    len_ratio = purified_len / original_len

    # 1.5배 이상 길어지거나 0.5배 이하로 짧아지면 이상
    if len_ratio > 1.7 or len_ratio < 0.5:
        logger.debug(
            "[OVER_MODIFIED] 길이 비율 이상 | ratio=%.2f | original=%d자 | purified=%d자",
            len_ratio,
            original_len,
            purified_len
        )
        return True

    return False


def check_meaning_preservation(original: str, purified: str) -> bool:
    """
    의미 보존 체크 (단어 겹침 기반)
    """
    # 간단한 단어 분리 (공백 기준)
    original_words = set(original.replace(',', ' ').replace('.', ' ').split())
    purified_words = set(purified.replace(',', ' ').replace('.', ' ').split())

    if not original_words:
        return True

    # 겹치는 단어 비율
    overlap = len(original_words & purified_words)
    similarity = overlap / len(original_words)

    # 핵심 단어가 40% 이상 유지되어야 함 (한국어는 조사 변화가 많아서 낮게 설정)
    threshold = 0.4

    if similarity < threshold:
        logger.debug(
            "[MEANING] 의미 유사도 낮음 | similarity=%.2f | threshold=%.2f",
            similarity,
            threshold
        )
        return False

    return True


def is_whitelisted(text: str) -> bool:
    """
    화이트리스트 체크 (순화 불필요 문장)
    """
    # 1-2학년이 자주 쓰는 순수 표현들
    WHITELIST = {
        "좋아", "싫어", "예뻐", "귀여워", "멋있어",
        "보고싶어", "사랑해", "맛있어", "재미있어",
        "고마워", "미안해", "반가워", "즐거워",
        "행복해", "신나", "기뻐", "슬퍼"
    }

    text_clean = text.strip().replace(' ', '')

    # 정확히 일치하는 경우
    if text_clean in WHITELIST:
        return True

    # 화이트리스트 단어만 포함하고 5자 이하인 경우
    if len(text_clean) <= 5:
        words = text_clean.split()
        if all(word in WHITELIST for word in words if word):
            return True

    return False


# =========================
# 고급 기능 (선택적 사용)
# =========================
def check_key_nouns_preserved(original: str, purified: str) -> bool:
    """
    주요 명사 보존 체크 (KoNLPy 필요 시 사용)
    현재는 비활성화 - 의존성 추가 필요
    """
    try:
        from konlpy.tag import Okt
        okt = Okt()

        original_nouns = [word for word, pos in okt.pos(original) if pos == 'Noun']
        purified_nouns = [word for word, pos in okt.pos(purified) if pos == 'Noun']

        # 주요 명사 2개 이상 보존되어야 함
        if len(original_nouns) >= 2:
            preserved_count = sum(1 for noun in original_nouns[:2] if noun in purified)
            if preserved_count < 1:
                return False

        return True
    except ImportError:
        logger.warning("[KEY_NOUNS] KoNLPy 미설치, 명사 체크 스킵")
        return True