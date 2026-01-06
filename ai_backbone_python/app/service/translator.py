import requests
import logging

logger = logging.getLogger("translator")

def translate_to_korean(text: str) -> str:
    """
    Google 비공식 번역 API를 사용한 영문 → 한글 번역
    - JS 코드와 동일한 로직
    - 실패 시 원문 반환
    """
    if not text:
        return ""

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "ko",
            "dt": "t",
            "q": text
        }

        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()

        # JS의 data?.[0]?.[0]?.[0] 대응
        if isinstance(data, list):
            return data[0][0][0] if data and data[0] and data[0][0] else text

        return text

    except Exception as e:
        logger.warning("Translation Error: %s", str(e))
        return text
