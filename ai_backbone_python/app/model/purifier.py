# app/model/purifier.py
import re
import torch
import logging
import sys
from app.model.model_loader import tokenizer, model, DEVICE, MAX_INPUT_LEN

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


def remove_dummy_tokens(text: str) -> str:
    return re.sub(r"(그그그|으으)", "", text or "")

def keep_before_first_period(text: str) -> str:
    if not text:
        return " "
    pos = text.find(".")
    return text if pos == -1 else text[:pos]

def purify_sentence(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
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

    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def refine(text: str) -> str:
    """
    ✅ Pchange=true일 때 호출되는 진입점
    - 모델 사용 여부를 로그로 명확히 남김
    - 순화 전 / 순화 후 문장을 모두 출력
    """
    print("🔥 purifier module loaded")
    original_text = text

    try:
        # 🔥 모델 통과
        purified = purify_sentence(text)

        # 🔧 후처리
        purified = remove_dummy_tokens(purified)
        purified = keep_before_first_period(purified)

        # ✅ 로그 출력 (핵심)
        logger.info(
            "[PURIFIER] model_used=True | before='%s' | after='%s'",
            original_text,
            purified
        )

        return purified

    except Exception as e:
        # ❌ 모델 미사용 (fallback)
        logger.warning(
            "[PURIFIER] model_used=False | before='%s' | reason=%s",
            original_text,
            str(e)
        )
        return original_text
# def refine(text: str) -> str:
#     try:
#         result = purify_sentence(text)
#         result = remove_dummy_tokens(result)
#         result = keep_before_first_period(result)
#         return result
#     except Exception:
#         return text
