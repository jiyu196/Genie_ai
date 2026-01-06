import os
from dotenv import load_dotenv
from pathlib import Path

# ai_backbone_python 디렉토리 기준으로 env.local 경로 계산
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / "env.local"

# 프로젝트 루트의 env.local 로드
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
