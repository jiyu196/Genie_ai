import csv
import random

ENCODING = "utf-8-sig"
random.seed(42)

# -----------------------------
# 설정
# -----------------------------
TYPE_CONFIG = {
    "A": {"file": r"D:\TEAMproject\KoBART_Training_Data\A_type.csv", "quota": 6000},
    "B": {"file": r"D:\TEAMproject\KoBART_Training_Data\B_type.csv", "quota": 6000},
    "C": {"file": r"D:\TEAMproject\KoBART_Training_Data\C_type.csv", "quota": 6000},
    "D": {"file": r"D:\TEAMproject\KoBART_Training_Data\D_type.csv", "quota": 4500},
}

INSTRUCTIONS = {
    "A": "[순화해줘: 장면 보존]\n- 등장인물 유지\n- 행동 유지\n- 장소 유지",
    "B": "[순화해줘: 공격성 완화]\n- 등장인물 유지\n- 행동은 중립적으로 완화\n- 장소 유지",
    "C": "[순화해줘: 어린이 친화 표현]\n- 의미 유지\n- 쉬운 말 사용\n- 부정적 뉘앙스 제거",
    "D": "[확인만 해줘]\n- 부적절한 표현 없음\n- 문장 그대로 유지",
}

OUTPUT_FILE = "../data/KoBART_input/train_ABCD_instruction_random_22500.csv"


# -----------------------------
def read_csv(path):
    with open(path, "r", encoding=ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "input_text", "target_text"]
        )
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
def build():
    # 1️⃣ 타입별 데이터 로딩 + 셔플
    data_pool = {}
    quota = {}

    for t, cfg in TYPE_CONFIG.items():
        rows = read_csv(cfg["file"])
        random.shuffle(rows)
        data_pool[t] = rows
        quota[t] = cfg["quota"]

    output = []
    new_id = 1

    # 2️⃣ 타입 선택 풀 생성 (quota 기반)
    type_bag = []
    for t, q in quota.items():
        type_bag.extend([t] * q)

    random.shuffle(type_bag)

    # 3️⃣ 랜덤 타입 순서대로 하나씩 추출
    for t in type_bag:
        if quota[t] <= 0:
            continue

        if not data_pool[t]:
            continue

        row = data_pool[t].pop()
        quota[t] -= 1

        input_text = f"{INSTRUCTIONS[t]}\n{row['forbidden_word'].strip()}"
        target_text = row["clean_word"].strip()

        output.append({
            "id": new_id,
            "input_text": input_text,
            "target_text": target_text,
        })
        new_id += 1

    write_csv(OUTPUT_FILE, output)
    print(f"[완료] 총 {len(output)}건 생성 → {OUTPUT_FILE}")


# -----------------------------
if __name__ == "__main__":
    build()
