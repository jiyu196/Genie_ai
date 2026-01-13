# app/model/model_loader.py
import os
import sys
import torch
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from contextlib import contextmanager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.core.config import MODEL_DIR

# 로거 설정
logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)

# =========================
# 설정
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "kobart_purifier_stage2_v4")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "checkpoint-5000")
MAX_INPUT_LEN = 128

# GPU 설정
CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))
ENABLE_HALF_PRECISION = os.getenv("ENABLE_HALF_PRECISION", "false").lower() == "true"


# =========================
# GPU/CPU 자동 감지
# =========================
def detect_device() -> str:
    """GPU 사용 가능 여부 확인 및 최적 디바이스 선택"""
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return "cpu"

    try:
        # 지정된 GPU 사용 가능 여부 확인
        if CUDA_DEVICE_ID >= torch.cuda.device_count():
            logger.warning(
                f"⚠️ CUDA 디바이스 {CUDA_DEVICE_ID}를 찾을 수 없습니다. "
                f"사용 가능한 디바이스: {torch.cuda.device_count()}"
            )
            return "cpu"

        device_name = torch.cuda.get_device_name(CUDA_DEVICE_ID)
        cuda_version = torch.version.cuda
        total_memory = torch.cuda.get_device_properties(CUDA_DEVICE_ID).total_memory / (1024 ** 3)

        logger.info(f"✅ GPU 감지: {device_name}")
        logger.info(f"   CUDA 버전: {cuda_version}")
        logger.info(f"   GPU 메모리: {total_memory:.2f} GB")
        logger.info(f"   디바이스 ID: {CUDA_DEVICE_ID}")

        return f"cuda:{CUDA_DEVICE_ID}"

    except Exception as e:
        logger.error(f"❌ GPU 정보 확인 실패: {e}")
        logger.warning("   CPU 모드로 전환합니다.")
        return "cpu"


DEVICE = detect_device()


# =========================
# 컨텍스트 매니저
# =========================
@contextmanager
def gpu_memory_manager():
    """GPU 메모리 관리 컨텍스트 매니저"""
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated(CUDA_DEVICE_ID) / (1024 ** 3)
        logger.debug(f"GPU 메모리 (시작): {start_mem:.2f} GB")

    try:
        yield
    finally:
        if "cuda" in DEVICE:
            torch.cuda.empty_cache()
            end_mem = torch.cuda.memory_allocated(CUDA_DEVICE_ID) / (1024 ** 3)
            logger.debug(f"GPU 메모리 (종료): {end_mem:.2f} GB")


# =========================
# 모델 경로 검증
# =========================
def validate_model_path(model_path: Path) -> Tuple[bool, str]:
    """모델 파일 존재 여부 및 필수 파일 검증"""
    if not model_path.exists():
        return False, f"모델 경로가 존재하지 않습니다: {model_path}"

    if not model_path.is_dir():
        return False, f"모델 경로가 디렉토리가 아닙니다: {model_path}"

    # 필수 파일 확인
    required_files = {
        "config.json": "모델 설정 파일",
        "tokenizer_config.json": "토크나이저 설정 파일",
    }

    # 모델 가중치 파일 (둘 중 하나 필요)
    has_weights = (model_path / "pytorch_model.bin").exists() or \
                  (model_path / "model.safetensors").exists()

    missing_files = []
    for file_name, description in required_files.items():
        if not (model_path / file_name).exists():
            missing_files.append(f"{file_name} ({description})")

    if not has_weights:
        missing_files.append("pytorch_model.bin 또는 model.safetensors (모델 가중치)")

    if missing_files:
        return False, f"필수 파일 누락: {', '.join(missing_files)}"

    return True, "모델 경로 검증 완료"


# =========================
# 모델 로딩
# =========================
def load_model_safe() -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSeq2SeqLM], str]:
    """
    안전한 모델 로딩

    Returns:
        (tokenizer, model, error_message)
    """
    MODEL_PATH = MODEL_DIR / MODEL_NAME / MODEL_CHECKPOINT

    logger.info("=" * 80)
    logger.info("🔄 모델 로딩 시작")
    logger.info(f"   경로: {MODEL_PATH}")
    logger.info(f"   디바이스: {DEVICE}")
    logger.info(f"   Half Precision: {ENABLE_HALF_PRECISION}")
    logger.info("=" * 80)

    start_time = time.time()

    # 1. 경로 검증
    is_valid, message = validate_model_path(MODEL_PATH)
    if not is_valid:
        error_msg = f"모델 경로 검증 실패: {message}"
        logger.critical(f"❌ {error_msg}")
        return None, None, error_msg

    try:
        with gpu_memory_manager():
            # 2. Tokenizer 로딩
            logger.info("📥 Tokenizer 로딩 중...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                trust_remote_code=False
            )
            logger.info("✅ Tokenizer 로딩 완료")

            # 3. Model 로딩
            logger.info("📥 Model 로딩 중...")

            load_kwargs = {
                "local_files_only": True,
                "trust_remote_code": False,
                "low_cpu_mem_usage": True
            }

            # Half precision 설정 (GPU only)
            if ENABLE_HALF_PRECISION and "cuda" in DEVICE:
                load_kwargs["torch_dtype"] = torch.float16
                logger.info("   Half precision (FP16) 활성화")

            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_PATH,
                **load_kwargs
            )
            logger.info("✅ Model 로딩 완료")

            # 4. 디바이스로 이동
            logger.info(f"🚀 Model을 {DEVICE}로 이동 중...")
            model.to(DEVICE)
            logger.info("✅ 디바이스 이동 완료")

            # 5. Evaluation 모드 설정
            model.eval()
            logger.info("✅ Model을 evaluation 모드로 설정")

            # 6. 메모리 최적화
            if "cuda" in DEVICE:
                # Gradient 계산 비활성화
                for param in model.parameters():
                    param.requires_grad = False

                # 메모리 사용량 확인
                allocated = torch.cuda.memory_allocated(CUDA_DEVICE_ID) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(CUDA_DEVICE_ID) / (1024 ** 3)
                logger.info(f"📊 GPU 메모리 사용량: {allocated:.2f} GB (예약: {reserved:.2f} GB)")

            load_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"✅ 모델 로딩 성공 (소요 시간: {load_time:.2f}초)")
            logger.info("=" * 80)

            return tokenizer, model, ""

    except FileNotFoundError as e:
        error_msg = f"모델 파일을 찾을 수 없습니다: {str(e)}"
        logger.critical(f"❌ {error_msg}")
        return None, None, error_msg

    except RuntimeError as e:
        error_msg = f"모델 로딩 중 런타임 오류: {str(e)}"
        logger.critical(f"❌ {error_msg}")

        if "out of memory" in str(e).lower():
            logger.critical("💥 GPU 메모리 부족!")
            logger.critical("   해결 방법:")
            logger.critical("   1. 다른 GPU 프로세스 종료")
            logger.critical("   2. ENABLE_HALF_PRECISION=true 설정")
            logger.critical("   3. 더 작은 배치 사이즈 사용")
            logger.critical("   4. CPU 모드로 실행")

        return None, None, error_msg

    except Exception as e:
        error_msg = f"예상치 못한 오류: {str(e)}"
        logger.critical(f"❌ {error_msg}")
        logger.exception("상세 오류:")
        return None, None, error_msg


# =========================
# 모델 로딩 실행 (Thread-Safe Lazy Loading)
# =========================
_init_lock = threading.Lock()
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None
_load_error: str = ""
_model_loaded: bool = False
_load_timestamp: Optional[float] = None


def init_model(force: bool = False) -> bool:
    """
    모델 초기화 (thread-safe)
    Args: force: True일 경우 이미 로딩된 모델을 재로딩
    Returns: 성공 여부
    """
    global _tokenizer, _model, _load_error, _model_loaded, _load_timestamp

    # Double-checked locking
    if _model_loaded and not force:
        return True

    with _init_lock:
        # Lock 획득 후 재확인
        if _model_loaded and not force:
            return True

        logger.info("🔄 모델 초기화 시작...")

        # 기존 모델 정리 (재로딩 시)
        if force and _model_loaded:
            logger.info("🔄 기존 모델 언로드 중...")
            cleanup_model()

        tok, mdl, err = load_model_safe()
        _tokenizer, _model, _load_error = tok, mdl, err
        _model_loaded = (_tokenizer is not None and _model is not None)
        _load_timestamp = time.time() if _model_loaded else None

        if not _model_loaded:
            logger.critical("=" * 80)
            logger.critical("⚠️⚠️⚠️ 모델 로딩 실패 ⚠️⚠️⚠️")
            logger.critical("서버는 시작되지만 프롬프트 정제 기능이 비활성화됩니다.")
            logger.critical(f"오류 원인: {_load_error}")
            logger.critical("=" * 80)
        else:
            logger.info("🎉 모델이 정상적으로 로드되었습니다.")

        return _model_loaded


def cleanup_model():
    """모델 메모리 정리"""
    global _tokenizer, _model, _model_loaded

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    _model_loaded = False

    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
        logger.info("GPU 메모리 정리 완료")


# =========================
# 모델 상태 확인 함수
# =========================
def is_model_available() -> bool:
    """모델 사용 가능 여부 확인"""
    return _model_loaded and _tokenizer is not None and _model is not None


def get_model_info() -> Dict:
    """모델 정보 반환"""
    info = {
        "model_name": MODEL_NAME,
        "checkpoint": MODEL_CHECKPOINT,
        "device": DEVICE,
        "loaded": _model_loaded,
        "max_input_len": MAX_INPUT_LEN,
        "half_precision": ENABLE_HALF_PRECISION,
        "error": _load_error if not _model_loaded else None,
    }

    if _load_timestamp:
        info["load_timestamp"] = _load_timestamp
        info["uptime_seconds"] = time.time() - _load_timestamp

    if _model_loaded and "cuda" in DEVICE:
        try:
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(CUDA_DEVICE_ID) / (1024 ** 3)
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(CUDA_DEVICE_ID) / (1024 ** 3)
        except:
            pass

    return info


def get_tokenizer() -> Optional[AutoTokenizer]:
    """토크나이저 반환 (내부용)"""
    return _tokenizer


def get_model() -> Optional[AutoModelForSeq2SeqLM]:
    """모델 반환 (내부용)"""
    return _model