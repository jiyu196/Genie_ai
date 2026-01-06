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
        # if isinstance(data, list) and data and isinstance(data[0], list):
        #     translated = "".join(
        #         segment[0] for segment in data[0]
        #         if isinstance(segment, list) and segment and segment[0]
        #     )
        #     return translated.strip() if translated else text
        if isinstance(data, list) and data and isinstance(data[0], list):
            # 🔑 핵심: 첫 문장 제외
            translated = "".join(
                segment[0]
                for segment in data[0][1:]
                if isinstance(segment, list) and segment and segment[0]
            )

            return translated.strip() if translated else ""
        return text

    except Exception as e:
        logger.warning("Translation Error: %s", str(e))
        return text
