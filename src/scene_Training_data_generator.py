import os
import json
import csv

# -----------------------------
# 경로 설정
# -----------------------------
INPUT_DIR = r"F:\Downloads\Sample\data\Training\TL_CA"
OUTPUT_CSV = "../data/KoBART_input/scene_description.csv"

# -----------------------------
# caption 키 목록
# -----------------------------
CAPTION_KEYS = [
    "caption_ko_1",
    "caption_ko_2",
    "caption_ko_3",
    "caption_ko_4",
    "caption_ko_5",
]

# -----------------------------
# 처리 시작
# -----------------------------
rows = []
auto_id = 1

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".json"):
        continue

    file_path = os.path.join(INPUT_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    context = data.get("context", {})

    for key in CAPTION_KEYS:
        sentence = context.get(key)

        # 방어 로직: None, 빈 문자열 제거
        if not sentence or not isinstance(sentence, str):
            continue

        rows.append([
            auto_id,
            sentence.strip()
        ])
        auto_id += 1

# -----------------------------
# CSV 저장
# -----------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "clean_word"])
    writer.writerows(rows)

print(f"완료: {OUTPUT_CSV}")
print(f"총 문장 수: {len(rows)}")
