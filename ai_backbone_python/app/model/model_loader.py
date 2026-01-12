# app/model/model_loader.py
import os
import sys
import torch
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.core.config import MODEL_DIR

# 로거 설정
logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)

# =========================
# 설정
# =========================
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "kobart_purifier_stage2_v4"
)

MODEL_CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT",
    "checkpoint-5000"
)

MAX_INPUT_LEN = 128


# =========================
# GPU/CPU 자동 감지
# =========================
def detect_device() -> str:
    """
    ✅ GPU 사용 가능 여부 확인
    ✅ CUDA 버전 로깅
    """
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        logger.warning("   GPU를 사용하려면 CUDA가 설치된 환경에서 실행하세요.")
        return "cpu"

    try:
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        logger.info(f"✅ GPU 감지: {device_name}")
        logger.info(f"   CUDA 버전: {cuda_version}")

        # GPU 메모리 확인
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"   GPU 메모리: {total_memory:.2f} GB")

        return "cuda"

    except Exception as e:
        logger.error(f"❌ GPU 정보 확인 실패: {e}")
        logger.warning("   CPU 모드로 전환합니다.")
        return "cpu"


DEVICE = detect_device()


# =========================
# 모델 경로 검증
# =========================
def validate_model_path(model_path: Path) -> Tuple[bool, str]:
    """
    ✅ 모델 파일 존재 여부 확인
    ✅ 필수 파일 검증
    """
    if not model_path.exists():
        return False, f"모델 경로가 존재하지 않습니다: {model_path}"

    if not model_path.is_dir():
        return False, f"모델 경로가 디렉토리가 아닙니다: {model_path}"

    # 필수 파일 확인
    required_files = [
        "config.json",
        "pytorch_model.bin",  # 또는 model.safetensors
        "tokenizer_config.json",
    ]

    missing_files = []
    for file_name in required_files:
        file_path = model_path / file_name
        alt_file_path = model_path / "model.safetensors"  # pytorch_model.bin 대신

        if file_name == "pytorch_model.bin":
            if not file_path.exists() and not alt_file_path.exists():
                missing_files.append(file_name)
        elif not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        return False, f"필수 파일 누락: {', '.join(missing_files)}"

    return True, "모델 경로 검증 완료"


# =========================
# 모델 로딩
# =========================
def load_model_safe() -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSeq2SeqLM], str]:
    """
    ✅ 안전한 모델 로딩
    ✅ 실패 시 None 반환 (서버 크래시 방지)

    Returns:
        (tokenizer, model, error_message)
    """
    MODEL_PATH = MODEL_DIR / MODEL_NAME / MODEL_CHECKPOINT

    logger.info("=" * 80)
    logger.info("🔄 모델 로딩 시작")
    logger.info(f"   경로: {MODEL_PATH}")
    logger.info(f"   디바이스: {DEVICE}")
    logger.info("=" * 80)

    # 1. 경로 검증
    is_valid, message = validate_model_path(MODEL_PATH)
    if not is_valid:
        error_msg = f"모델 경로 검증 실패: {message}"
        logger.critical(f"❌ {error_msg}")
        return None, None, error_msg

    try:
        # 2. Tokenizer 로딩
        logger.info("📥 Tokenizer 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False  # 보안을 위해 False
        )
        logger.info("✅ Tokenizer 로딩 완료")

        # 3. Model 로딩
        logger.info("📥 Model 로딩 중...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False,
            low_cpu_mem_usage=True  # 메모리 효율적 로딩
        )
        logger.info("✅ Model 로딩 완료")

        # 4. GPU로 이동
        if DEVICE == "cuda":
            logger.info("🚀 Model을 GPU로 이동 중...")
            model.to(DEVICE)
            logger.info("✅ GPU 이동 완료")
        else:
            logger.info("⚠️ CPU 모드로 실행")
            model.to(DEVICE)

        # 5. Evaluation 모드 설정
        model.eval()
        logger.info("✅ Model을 evaluation 모드로 설정")

        # 6. 메모리 사용량 확인
        if DEVICE == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
            logger.info(f"📊 GPU 메모리 사용량: {allocated:.2f} GB (예약: {reserved:.2f} GB)")

        logger.info("=" * 80)
        logger.info("✅ 모델 로딩 성공")
        logger.info("=" * 80)

        return tokenizer, model, ""

    except FileNotFoundError as e:
        error_msg = f"모델 파일을 찾을 수 없습니다: {str(e)}"
        logger.critical(f"❌ {error_msg}")
        return None, None, error_msg

    except RuntimeError as e:
        error_msg = f"모델 로딩 중 런타임 오류: {str(e)}"
        logger.critical(f"❌ {error_msg}")

        # GPU 메모리 부족 감지
        if "out of memory" in str(e).lower():
            logger.critical("💥 GPU 메모리 부족!")
            logger.critical("   해결 방법:")
            logger.critical("   1. 다른 GPU 프로세스 종료")
            logger.critical("   2. 더 작은 모델 사용")
            logger.critical("   3. CPU 모드로 실행")

        return None, None, error_msg

    except Exception as e:
        error_msg = f"예상치 못한 오류: {str(e)}"
        logger.critical(f"❌ {error_msg}")
        logger.exception("상세 오류:")
        return None, None, error_msg


# =========================
#  모델 로딩 실행
# Lazy load + thread-safe init
# - import 시점에 로딩하지 않음 (uvicorn reload/worker 환경에서 예측 가능성 ↑)
# - 최초 요청 또는 startup에서 init_model()을 호출해 1회만 로딩
# =========================
_init_lock = threading.Lock()
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None
_load_error: str = ""
_model_loaded: bool = False

def init_model(force: bool = False) -> bool:
    """
+    ✅ 모델 초기화 (thread-safe)
+    - force=False: 이미 로딩되었으면 스킵
+    - force=True: 강제로 재로딩 시도 (운영에서는 비권장)
+    """
    global _tokenizer, _model, _load_error, _model_loaded

    if _model_loaded and not force:
        return True

    with _init_lock:
        if _model_loaded and not force:
            return True

        tok, mdl, err = load_model_safe()
        _tokenizer, _model, _load_error = tok, mdl, err
        _model_loaded = (_tokenizer is not None and _model is not None)

        if not _model_loaded:
            logger.critical("=" * 80)
            logger.critical("⚠️⚠️⚠️ 모델 로딩 실패 ⚠️⚠️⚠️")
            logger.critical("서버는 시작되지만 프롬프트 정제 기능이 비활성화됩니다.")
            logger.critical(f"오류 원인: {_load_error}")
            logger.critical("=" * 80)
        else:
            logger.info("🎉 모델이 정상적으로 로드되었습니다.")

        return _model_loaded

#tokenizer, model, load_error = load_model_safe()
# 로딩 실패 시 경고 (서버는 시작하되 기능 제한)
# if tokenizer is None or model is None:
#     logger.critical("=" * 80)
#     logger.critical("⚠️⚠️⚠️ 모델 로딩 실패 ⚠️⚠️⚠️")
#     logger.critical("서버는 시작되지만 프롬프트 정제 기능이 비활성화됩니다.")
#     logger.critical(f"오류 원인: {load_error}")
#     logger.critical("=" * 80)
#
#     # Fallback: 더미 객체 (타입 체크용)
#     MODEL_LOADED = False
# else:
#     MODEL_LOADED = True
#     logger.info("🎉 모델이 정상적으로 로드되었습니다.")



# =========================
# 모델 상태 확인 함수
# =========================
def is_model_available() -> bool:
    """모델 사용 가능 여부 확인"""
#    return MODEL_LOADED and tokenizer is not None and model is not None
    return _model_loaded and _tokenizer is not None and _model is not None


def get_model_info() -> dict:
    """모델 정보 반환"""
    return {
        "model_name": MODEL_NAME,
        "checkpoint": MODEL_CHECKPOINT,
        "device": DEVICE,
#        "loaded": MODEL_LOADED,
        "loaded": _model_loaded,
        "max_input_len": MAX_INPUT_LEN,
#        "error": load_error if not MODEL_LOADED else None
        "error": _load_error if not _model_loaded else None
    }