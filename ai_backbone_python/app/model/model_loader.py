# model_loader.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.core.config import MODEL_DIR

# 어떤 모델을 쓸지 선택 (핵심)
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "kobart_purifier_stage2_v4"
#    "kobart_purifier_stage3_short"
)

# 체크포인트 선택
MODEL_CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT",
    "checkpoint-5000"
#    "checkpoint-2000"
)

MODEL_PATH = MODEL_DIR / MODEL_NAME / MODEL_CHECKPOINT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

MAX_INPUT_LEN = 128
