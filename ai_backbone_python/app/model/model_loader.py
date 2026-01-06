# model_loader.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(
    BASE_DIR,
    "kobart_purifier_stage2_v4",
    "checkpoint-5000"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")



tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

MAX_INPUT_LEN = 128
