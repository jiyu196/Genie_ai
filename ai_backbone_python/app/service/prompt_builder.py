# app/service/prompt_builder.py
"""
웹툰 스타일 이미지 생성을 위한 프롬프트 구성 모듈
"""
import re
import logging
import requests
from typing import Optional

logger = logging.getLogger("prompt_builder")

# 우리가 보내는 스타일 키워드 (정확히 이것만 제거)
STYLE_KEYWORDS_TO_REMOVE = [
    "manhwa-inspired composition",
    # "Korean webtoon style illustration",
    "clean lines",
    "vibrant colors",
    "professional digital illustration",
    "studio quality",
    "manhwa art style",
    "digital art"
    # "Korean webtoon panel style",
    "vertical scroll comic style",
     "soft pastel color palette",

    # "natural shading",
    # "expressive character faces",
    # "subtle line art",
    # "clean but not vectorized",
    # "dramatic lighting",
    # "cinematic framing",
    # "emotional atmosphere",
    # "slice-of-life manhwa style"
    # "modern webtoon panel style",
    # "soft pastel color palette",
    # "natural cel shading",
    # "expressive character faces with detailed eyes",
    # "clean line art with subtle variation",
    # "digital illustration style",
    # "dramatic lighting and depth",
    # "cinematic vertical framing",
    # "emotional atmosphere",
    # "contemporary romance comic aesthetic",
    # "manhwa-inspired composition"

    # "vertical scroll comic style",
    # "soft pastel colors with gradient shading",
    # "clean lines",
    # "vibrant colors",
    # "natural cel shading",
    # "expressive faces with detailed emotional eyes",
    # "clean digital line art",
    # "cinematic panel composition",
    # "dramatic mood lighting",
    # "romance illustration aesthetic",
    # "emotional atmosphere",
    # "cel-shaded rendering"
]

def ko_to_ko_translate(text: str) -> str:
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "ko",
            "tl": "ko",
            "dt": "t",
            "q": text
        }

        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()

        # Google Translate 응답 구조
        translated = "".join([seg[0] for seg in data[0] if seg[0]])
        return translated.strip()

    except Exception:
        # 실패 시 원본 반환 (절대 깨지지 않게)
        return text

# =========================
# ⭐ 사용자 응답용 (신규 핵심)
# =========================
def compose_korean_scene(
    character_description: Optional[str],
    scene_description: str
) -> str:
    scene = scene_description.strip()

    if character_description:
        character = character_description.strip()
        text = f"{character}의 모습이 담긴 장면으로, {scene}"
    else:
        text = scene

    return ko_to_ko_translate(text)
def remove_style_from_revised_prompt(revised_prompt: str) -> str:
    """
    revised_prompt에서 우리가 추가한 스타일 키워드만 간단히 제거

    Args:
        revised_prompt: DALL-E의 revised_prompt (영문)

    Returns:
        스타일이 제거된 텍스트
    """
    if not revised_prompt:
        return ""

    text = revised_prompt

    # 1. 우리가 보낸 스타일 키워드를 하나씩 제거
    for keyword in STYLE_KEYWORDS_TO_REMOVE:
        # 대소문자 무시하고 제거
        pattern = re.escape(keyword)
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 2. 추가로 자주 나오는 스타일 관련 접두사 제거
    # style_prefixes = [
    #     r"^Create\s+an?\s+",
    #     r"^An?\s+illustration\s+",
    #     r"^The\s+image\s+",
    #     r"resembling\s+",
    #     r"in\s+the\s+style\s+of\s+",
    #     r"featuring\s+",
    #     r"depicting\s+",
    #     r"with\s+",
    # ]
    style_prefixes = [
        r"^Create\s+an?\s+",
        r"^An?\s+illustration\s+",
        r"resembling\s+",
        r"in\s+the\s+style\s+of\s+",
    ]
    for prefix in style_prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # 3. 공백과 구두점 정리
    text = re.sub(r"\s{2,}", " ", text)  # 연속 공백 제거
    text = re.sub(r"\s*,\s*,\s*", ", ", text)  # 중복 쉼표 제거
    text = re.sub(r"^\s*,\s*", "", text)  # 앞쪽 쉼표 제거
    text = re.sub(r"\s*,\s*$", "", text)  # 뒤쪽 쉼표 제거
    text = re.sub(r"^\.+\s*", "", text)  # 앞쪽 마침표 제거

    # 4. 앞뒤 공백 제거
    text = text.strip()

    # 5. 첫 글자 대문자로 (문장 시작)
    if text:
        text = text[0].upper() + text[1:]

    return text


def build_webtoon_prompt(
        character_description: Optional[str],
        scene_description: str,
        include_style: bool = True
) -> str:
    """
    웹툰 스타일 이미지를 위한 구조화된 프롬프트 생성

    Args:
        character_description: 캐릭터 설명 (예: "파란 머리의 소녀, 큰 눈, 학교 교복")
        scene_description: 장면/동작 설명 (예: "캐릭터가 웃고 있어")
        include_style: 스타일 정보 포함 여부

    Returns:
        구조화된 완전한 프롬프트
    """

    prompt_parts = []

    # 1. 스타일 정의 (맨 앞에 배치) - 영문
    if include_style:
        style_text = ", ".join(STYLE_KEYWORDS_TO_REMOVE[:5])  # 처음 5개만 사용
        prompt_parts.append(style_text)
        logger.debug("[PROMPT_BUILDER] Style added: %s", style_text)

    # 2. 캐릭터 설명 - 한글
    if character_description and character_description.strip():
        character_part = f"Character: {character_description.strip()}"
        prompt_parts.append(character_part)
        logger.debug("[PROMPT_BUILDER] Character added: %s", character_part)

    # 3. 장면/동작 설명 - 한글
    if scene_description and scene_description.strip():
        scene_part = f"Scene: {scene_description.strip()}"
        prompt_parts.append(scene_part)
        logger.debug("[PROMPT_BUILDER] Scene added: %s", scene_part)

    # 4. 모든 파트를 결합
    final_prompt = ". ".join(prompt_parts) + "."

    return final_prompt


def build_detailed_webtoon_prompt(
        character_description: Optional[str],
        scene_description: str,
        mood: Optional[str] = None,
        background: Optional[str] = None,
        lighting: Optional[str] = None
) -> str:
    """
    더 세밀한 제어를 위한 상세 프롬프트 생성

    Args:
        character_description: 캐릭터 설명
        scene_description: 장면/동작 설명
        mood: 분위기 (예: "cheerful", "dramatic", "melancholic")
        background: 배경 설명 (예: "school classroom", "city street")
        lighting: 조명 (예: "bright daylight", "soft evening light")

    Returns:
        상세 구조화된 프롬프트
    """

    prompt_sections = []

    # 1. 스타일
    style_text = ", ".join(STYLE_KEYWORDS_TO_REMOVE[:3])
    prompt_sections.append(style_text)

    # 2. 캐릭터
    if character_description:
        prompt_sections.append(f"Character: {character_description.strip()}")

    # 3. 장면/동작
    if scene_description:
        prompt_sections.append(f"Action: {scene_description.strip()}")

    # 4. 배경
    if background:
        prompt_sections.append(f"Background: {background.strip()}")

    # 5. 분위기
    if mood:
        prompt_sections.append(f"Mood: {mood.strip()}")

    # 6. 조명
    if lighting:
        prompt_sections.append(f"Lighting: {lighting.strip()}")

    final_prompt = ". ".join(prompt_sections) + "."

    return final_prompt


def log_prompt_construction(
        access_id: str,
        original_prompt: str,
        character_description: Optional[str],
        final_prompt: str
) -> None:
    """
    프롬프트 구성 과정을 콘솔에 상세히 출력

    Args:
        access_id: 사용자 ID
        original_prompt: 원본 프롬프트
        character_description: 캐릭터 설명
        final_prompt: 최종 구성된 프롬프트
    """

    separator = "=" * 80

    print("\n" + separator)
    print("🎨 WEBTOON PROMPT CONSTRUCTION")
    print(separator)
    print(f"📍 Access ID: {access_id}")
    print(separator)

    # 원본 프롬프트
    print("\n📝 [ORIGINAL PROMPT]")
    print(f"   {original_prompt}")

    # 캐릭터 정보
    if character_description:
        print("\n👤 [CHARACTER DESCRIPTION]")
        print(f"   {character_description}")
    else:
        print("\n👤 [CHARACTER DESCRIPTION]")
        print("   (No character saved)")

    # 최종 프롬프트
    print("\n✨ [FINAL CONSTRUCTED PROMPT]")
    print(f"   {final_prompt}")

    # 통계
    # print("\n📊 [STATISTICS]")
    # print(f"   Original length: {len(original_prompt)} chars")
    # if character_description:
    #     print(f"   Character length: {len(character_description)} chars")
    # print(f"   Final length: {len(final_prompt)} chars")
    # print(f"   Total tokens (approx): {len(final_prompt.split())} words")

    print(separator)
    print("🚀 Sending to DALL-E...")
    print(separator + "\n")

    # 로거에도 기록
    logger.info(
        "[PROMPT_CONSTRUCTION] access_id=%s | original='%s' | character='%s' | final='%s'",
        access_id,
        original_prompt[:100],
        character_description[:100] if character_description else "None",
        final_prompt[:150]
    )


def log_revised_prompt_cleaning(
        access_id: str,
        original_revised: str,
        cleaned_revised: str
) -> None:
    """
    revised_prompt 정리 과정을 콘솔에 출력
    """
    separator = "=" * 80

    print("\n" + separator)
    print("🧹 CLEANING REVISED PROMPT")
    print(separator)
    print(f"📍 Access ID: {access_id}")
    print(separator)

    print("\n📥 [DALL-E REVISED PROMPT] (Original)")
    print(f"   {original_revised}")

    print("\n🧹 [AFTER STYLE REMOVAL]")
    print(f"   {cleaned_revised}")

    print("\n📊 [STATISTICS]")
    print(f"   Original: {len(original_revised)} chars")
    print(f"   Cleaned: {len(cleaned_revised)} chars")
    print(f"   Removed: {len(original_revised) - len(cleaned_revised)} chars")

    print(separator)
    print("🌐 Ready for translation...")
    print(separator + "\n")

    # 로거에도 기록
    logger.info(
        "[REVISED_CLEANING] access_id=%s | original_len=%d | cleaned_len=%d | removed=%d",
        access_id,
        len(original_revised),
        len(cleaned_revised),
        len(original_revised) - len(cleaned_revised)
    )