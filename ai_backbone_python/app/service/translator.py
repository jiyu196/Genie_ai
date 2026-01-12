# app/service/translator.py
import aiohttp
import asyncio
import requests
import logging
from typing import Optional

logger = logging.getLogger("translator")


async def translate_to_korean_async(text: str) -> str:
    """
    비동기 영문 → 한글 번역 (Google 번역 API)

    Args:
        text: 번역할 영문 텍스트

    Returns:
        번역된 한글 텍스트 (실패 시 원문)
    """
    if not text or not text.strip():
        logger.debug("[TRANSLATE_ASYNC] 빈 입력")
        return ""

    text = text.strip()

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",  # source language: English
            "tl": "ko",  # target language: Korean
            "dt": "t",  # return translation
            "q": text
        }

        logger.info(
            "[TRANSLATE_ASYNC] 번역 시작 | text_len=%d | preview=%s...",
            len(text), text[:50]
        )

        # aiohttp로 비동기 HTTP 요청
        timeout = aiohttp.ClientTimeout(total=5)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                # 응답 구조 검증
                if not isinstance(data, list) or not data:
                    logger.warning("[TRANSLATE_ASYNC] 응답 구조 이상 (1) | type=%s", type(data).__name__)
                    return text

                if not isinstance(data[0], list):
                    logger.warning("[TRANSLATE_ASYNC] 응답 구조 이상 (2) | type=%s", type(data[0]).__name__)
                    return text

                # 번역 결과 추출
                # 응답 구조: [[["번역1", "원문1", ...], ["번역2", "원문2", ...]], ...]
                # 첫 문장 제외하고 결합 (첫 문장은 원문 반복)
                translated_parts = []

                for segment in data[0][1:]:  # 첫 번째 세그먼트 제외
                    if isinstance(segment, list) and len(segment) > 0:
                        part = segment[0]
                        if part and isinstance(part, str):
                            translated_parts.append(part)

                if translated_parts:
                    result = "".join(translated_parts).strip()

                    logger.info(
                        "[TRANSLATE_ASYNC] 번역 성공 | original_len=%d | translated_len=%d",
                        len(text), len(result)
                    )

                    return result if result else text
                else:
                    logger.warning("[TRANSLATE_ASYNC] 번역 결과 없음, 원문 반환")
                    return text

    except aiohttp.ClientError as e:
        logger.warning("[TRANSLATE_ASYNC] 네트워크 오류: %s | 원문 반환", str(e))
        return text

    except asyncio.TimeoutError:
        logger.warning("[TRANSLATE_ASYNC] 타임아웃 | 원문 반환")
        return text

    except Exception as e:
        logger.error("[TRANSLATE_ASYNC] 예상치 못한 오류: %s | 원문 반환", str(e))
        return text


def translate_to_korean(text: str) -> str:
    """
    동기 버전 영문 → 한글 번역 (하위 호환용)

    Args:
        text: 번역할 영문 텍스트

    Returns:
        번역된 한글 텍스트 (실패 시 원문)
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "ko",
            "dt": "t",
            "q": text
        }

        logger.info(
            "[TRANSLATE] 번역 시작 (동기) | text_len=%d",
            len(text)
        )

        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()

        # 응답 구조 검증
        if isinstance(data, list) and data and isinstance(data[0], list):
            # 첫 문장 제외하고 결합
            translated_parts = []

            for segment in data[0][1:]:
                if isinstance(segment, list) and len(segment) > 0:
                    part = segment[0]
                    if part and isinstance(part, str):
                        translated_parts.append(part)

            if translated_parts:
                result = "".join(translated_parts).strip()

                logger.info(
                    "[TRANSLATE] 번역 성공 (동기) | translated_len=%d",
                    len(result)
                )

                return result if result else text

        logger.warning("[TRANSLATE] 번역 결과 없음, 원문 반환")
        return text

    except requests.RequestException as e:
        logger.warning("[TRANSLATE] 네트워크 오류: %s | 원문 반환", str(e))
        return text

    except Exception as e:
        logger.error("[TRANSLATE] 예상치 못한 오류: %s | 원문 반환", str(e))
        return text


# =========================
# 테스트/디버그용 함수
# =========================
async def test_translate_async():
    """비동기 번역 테스트"""
    test_cases = [
        "A blue-haired girl with big eyes",
        "Beautiful sunset over the ocean",
        "Futuristic robot in action",
        "",  # 빈 문자열
        "   ",  # 공백만
    ]

    print("\n" + "=" * 80)
    print("🧪 비동기 번역 테스트")
    print("=" * 80)

    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"  원문: '{text}'")

        result = await translate_to_korean_async(text)

        print(f"  번역: '{result}'")
        print(f"  변경: {text != result}")

    print("\n" + "=" * 80)


def test_translate_sync():
    """동기 번역 테스트"""
    test_cases = [
        "A blue-haired girl with big eyes",
        "Beautiful sunset over the ocean",
    ]

    print("\n" + "=" * 80)
    print("🧪 동기 번역 테스트")
    print("=" * 80)

    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"  원문: '{text}'")

        result = translate_to_korean(text)

        print(f"  번역: '{result}'")
        print(f"  변경: {text != result}")

    print("\n" + "=" * 80)


# 직접 실행 시 테스트
if __name__ == "__main__":
    # 동기 버전 테스트
    test_translate_sync()

    # 비동기 버전 테스트
    print("\n")
    asyncio.run(test_translate_async())