"""
⚠️ DEPRECATED: 이 모듈은 더 이상 사용되지 않습니다.
대신 app.model.validation을 사용하세요.

후처리 모듈: KoBART 모델 출력 검증 및 보정 (Optimized)
"""

import re
import logging

logger = logging.getLogger("post_processor")

# =========================
# 🔹 정규식 / 상수 캐시
# =========================

RE_REPEAT = re.compile(r'(.)\1{4,}')
RE_HAS_TEXT = re.compile(r'[가-힣a-zA-Z]')

INVALID_TOKENS = ('<unk>', '[PAD]', '[UNK]', '<pad>', '<s>', '</s>')

PUNCT_TRANS = str.maketrans({',': ' ', '.': ' '})

WHITELIST = {
    "좋아", "싫어", "예뻐", "귀여워", "멋있어",
    "보고싶어", "사랑해", "맛있어", "재미있어",
    "고마워", "미안해", "반가워", "즐거워",
    "행복해", "신나", "기뻐", "슬퍼"
}

# =========================
# 🔹 Main Pipeline
# =========================

def post_process(original: str, purified: str) -> str:

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "[POST_PROCESS] start | original=%s | purified=%s",
            original[:30], purified[:30]
        )

    if not validate_output(purified):
        return original

    if is_over_modified(original, purified):
        return original

    if not check_meaning_preservation(original, purified):
        return original

    # if is_whitelisted(original):
    #     return original

    return purified.strip()


# =========================
# 🔹 Validation
# =========================

def validate_output(purified: str) -> bool:
    if not purified:
        return False

    p = purified.strip()

    if len(p) < 2:
        return False

    if RE_REPEAT.search(p):
        return False

    if not RE_HAS_TEXT.search(p):
        return False

    for token in INVALID_TOKENS:
        if token in p:
            return False

    return True


# =========================
# 🔹 Over-modification
# =========================

def is_over_modified(original: str, purified: str) -> bool:
    o = len(original.strip())
    p = len(purified.strip())

    if o == 0:
        return False

    ratio = p / o
    return ratio > 1.7 or ratio < 0.5


# =========================
# 🔹 Meaning preservation
# =========================

def check_meaning_preservation(original: str, purified: str) -> bool:
    o_words = original.translate(PUNCT_TRANS).split()
    if not o_words:
        return True

    p_set = set(purified.translate(PUNCT_TRANS).split())

    need = max(1, int(len(o_words) * 0.4))
    hit = 0

    for w in o_words:
        if w in p_set:
            hit += 1
            if hit >= need:
                return True

    return False


# =========================
# 🔹 Whitelist
# =========================

def is_whitelisted(text: str) -> bool:
    t = text.replace(" ", "").strip()

    if t in WHITELIST:
        return True

    if len(t) <= 5:
        temp = t
        for w in WHITELIST:
            temp = temp.replace(w, "")
        return not temp

    return False
