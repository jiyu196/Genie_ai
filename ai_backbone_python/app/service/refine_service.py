# app/service/refine_service.py
from app.model.purifier import refine

def process_prompt(origin_prompt: str, is_slang: bool) -> dict:
    """
    is_slang 값에 따라 모델 실행 여부 분기
    """
    if is_slang:
        revised = refine(origin_prompt)
        return {
            "filtered_content": revised,
            "refined_content": revised
        }
    else:
        return {
            "filtered_content": origin_prompt,
            "refined_content": origin_prompt
        }

def refine_prompt(prompt: str) -> dict:
    refined = refine(prompt)
    model_used = refined != prompt

    return {
        "original_content": prompt,
        "refined_content": refined,
        "is_slang": model_used,
    }
