import pandas as pd

src_csv = "../data/replace_puriword/purified_words.csv"        # 원본 테이블
out_csv = "../data/KoBART_input/koBART_train.csv"           # 학습용

df = pd.read_csv(src_csv)

train_df = pd.DataFrame({
    "input_text": "순화해줘: " + df["forbidden_word"],
    "target_text": df["clean_word"]
})

train_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("학습 데이터 생성 완료")
