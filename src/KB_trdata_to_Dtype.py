# ==========================================================
# 데이터에 랜덤규칙으로 초등학생이 입력한것 같은 학습데이터를 만든다
# ==========================================================
import pandas as pd
import random
import re

JOSA_PATTERN = r"(에서|에게|으로|를|을|은|는|이|가|도|만|밖에|뿐|보다|커녕|마저|조차|에|로|까지|부터|와|과|랑|이랑|하고)"
INTENSIFIERS = ["너무", "되게", "엄청", "진짜","정말", "왕창", "후덜덜", "어", "앗",
                "완전", "살짝", "약간", "좀", "아주", "꽤", "왠지", "진짜로", "매우", "엄청나게", "살짝", "몰래", "갑자기",
                "자주", "가끔", "맨날", "항상", "종종", "거의",
                "먼저", "나중에", "곧", "드디어", "벌써", "갑자기", "드문드문",
                "빨리", "천천히", "살금살금", "조용히", "살짝살짝", "씩씩하게", "용감하게"
]
CONNECTORS = ["그리고", "그래서","하지만", "또는", "왜냐하면","그러나",
              "그러고", "그다음에", "그래서", "그러다가", "그런데", "그러면",
              "왜냐하면", "그러니까", "따라서",
              "하지만", "근데", "그렇지만",
              "아니면", "또", "게다가"
]
ERROR_TYPES = [
    "JOSA_DROP",
    "INTENSIFIER",
    "CONNECTOR_REPEAT",
    "ENDING_SIMPLE",
    "WORD_REPEAT"
]

ERROR_WEIGHTS = {
    "JOSA_DROP": 0.35,
    "INTENSIFIER": 0.30,
    "CONNECTOR_REPEAT": 0.15,
    "ENDING_SIMPLE": 0.15,
    "WORD_REPEAT": 0.05
}

# def make_childish_sentence(sentence: str) -> str:
#     s = sentence.strip()
#
#     if not s:
#         return s
#
#     # --- 1️⃣ 조사 오류 (누락 또는 중복) ---
#     if random.random() < 0.6:
#         s = re.sub(JOSA_PATTERN, "", s, count=random.randint(1, 2))
#     elif random.random() < 0.2:
#         s = re.sub(JOSA_PATTERN, r"\1\1", s, count=1)
#
#     # --- 2️⃣ 강조어 과잉 ---
#     if random.random() < 0.5:
#         prefix = random.choice(INTENSIFIERS)
#         if not s.startswith(prefix):
#             s = f"{prefix} {s}"
#
#     # --- 3️⃣ 연결어 미숙 사용 ---
#     if random.random() < 0.4:
#         for c in CONNECTORS:
#             if c in s:
#                 s = s.replace(c, f"{c} {c}", 1)
#                 break
#
#     # --- 4️⃣ 어순 단순화 (앞부분 반복) ---
#     if random.random() < 0.3:
#         words = s.split()
#         if len(words) > 4:
#             s = " ".join(words[:2]) + " " + s
#
#     # --- 5️⃣ 종결어미 단순화 ---
#     if random.random() < 0.5:
#         s = re.sub(r"(입니다|있습니다|합니다|됩니다)$", "해요", s)
#
#     # --- 6️⃣ 문장 미완성 느낌 ---
#     if random.random() < 0.3:
#         s = s.rstrip(".")
#     if random.random() < 0.2:
#         s += "…"
#
#     return s

def make_childish_sentence(sentence: str) -> str:
    s = sentence.strip()
    if not s:
        return s

    # ✅ 문장당 오류 개수 제한
    error_count = random.choices([1, 2], weights=[0.7, 0.3])[0]

    error_types = random.choices(
        population=list(ERROR_WEIGHTS.keys()),
        weights=list(ERROR_WEIGHTS.values()),
        k=error_count
    )

    for err in error_types:
        if err == "JOSA_DROP":
            s = re.sub(JOSA_PATTERN, "", s, count=1)

        elif err == "INTENSIFIER":
            s = random.choice(INTENSIFIERS) + " " + s

        elif err == "CONNECTOR_REPEAT":
            for c in CONNECTORS:
                if c in s:
                    s = s.replace(c, f"{c} {c}", 1)
                    break

        elif err == "ENDING_SIMPLE":
            s = re.sub(r"(입니다|있습니다|합니다|됩니다)$", "해요", s)

        elif err == "WORD_REPEAT":
            words = s.split()
            if len(words) > 3:
                s = words[0] + " " + words[0] + " " + " ".join(words[1:])

    return s
# ---------- CSV 처리 ----------
CSV_PATH = "../data/KoBART_input/scene_description.csv"
#OUT_PATH = "../data/KoBART_input/scene_description_training.csv"
TEST_CSV_PATH = "../data/KoBART_input/test.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

if "forbidden_word" not in df.columns:
    df["forbidden_word"] = ""

for idx, row in df.iterrows():
    clean_sentence = str(row["clean_word"]).strip()

    if not clean_sentence:
        continue

    df.at[idx, "forbidden_word"] = make_childish_sentence(clean_sentence)
# 원본 CSV 덮어쓰기
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

# 동일 데이터 test.csv 저장
df.to_csv(TEST_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"- 원본 업데이트: {CSV_PATH}")
print(f"- 테스트 파일 생성: {TEST_CSV_PATH}")
print(f"총 처리 문장 수: {len(df)}")
