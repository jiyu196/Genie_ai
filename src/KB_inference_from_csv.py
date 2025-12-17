import os
import time
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# =============================
# 1. 환경 / 모델 로드
# =============================
MODEL_NAME = "gogamza/kobart-base-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device 사용 중: {device}")

if device.type == "cuda":
    print(f"[INFO] GPU 이름: {torch.cuda.get_device_name(0)}")

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# =============================
# 2. 문장 순화 (테스트용)
# =============================
def refine_sentence_test(text: str) -> str:
    """
    ⚠ 테스트 목적:
    - 품질 보장 X
    - KoBART 기본 성향 확인
    """
    prompt = f"rewrite: {text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    # BART는 token_type_ids 사용 안 함
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            do_sample=False,          # 랜덤성 제거 (테스트 안정성)
            no_repeat_ngram_size=3,   # 반복 폭주 최소 억제
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =============================
# 3. CSV 로드
# =============================
input_csv_path = "../data/KoBART_input/koBART_test.csv"
output_csv_path = "../data/KoBART_input/refined_result_test.csv"

if not os.path.exists(input_csv_path) or os.path.getsize(input_csv_path) == 0:
    raise ValueError("입력 CSV 파일이 비어 있거나 존재하지 않습니다.")

df = pd.read_csv(input_csv_path)

# =============================
# 4. 테스트 실행 + 시간 측정
# =============================
results = []

print("\n[INFO] KoBART 테스트 추론 시작\n")

for _, row in df.iterrows():
    start = time.perf_counter()

    refined = refine_sentence_test(row["input_text"])

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"[ID {row['id']}] 입력: {row['input_text']}")
    print(f"[ID {row['id']}] 출력: {refined}")
    print(f"[ID {row['id']}] 소요 시간: {elapsed_ms:.2f} ms\n")

    results.append({
        "id": row["id"],
        "input_text": row["input_text"],
        "generated_text": refined,
        "elapsed_ms": round(elapsed_ms, 2)
    })

# =============================
# 5. 결과 저장
# =============================
out_df = pd.DataFrame(results)
out_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"[INFO] 테스트 완료 → {output_csv_path}")
