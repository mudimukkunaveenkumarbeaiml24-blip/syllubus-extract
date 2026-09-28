import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD .ENV
# ============================================================

from dotenv import dotenv_values

ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)
ENV_VALUES = dotenv_values(ENV_FILE)

NVIDIA_API_KEY = (ENV_VALUES.get("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY", "")).strip()

NVIDIA_BASE_URL = (ENV_VALUES.get("NVIDIA_BASE_URL") or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")).strip()

NVIDIA_MODEL = (ENV_VALUES.get("NVIDIA_MODEL") or os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")).strip()

NVIDIA_BACKUP_API_KEY = (ENV_VALUES.get("NVIDIA_BACKUP_API_KEY") or os.getenv("NVIDIA_BACKUP_API_KEY", "")).strip()

NVIDIA_BACKUP_MODEL = (ENV_VALUES.get("NVIDIA_BACKUP_MODEL") or os.getenv("NVIDIA_BACKUP_MODEL", "meta/muse-glimmer-30b")).strip()


# ============================================================
# MONGODB
# ============================================================

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017"
).strip()

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "syllabusiq"
).strip()


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR = BASE_DIR / "outputs"

TEMPLATES_DIR = BASE_DIR / "templates"


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXTRACTION SETTINGS
# ============================================================

MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "100"))

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))


# ============================================================
# OCR SETTINGS
# ============================================================

OCR_ENABLED = os.getenv("OCR_ENABLED", "True").lower() in ("true", "1", "yes")

OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng").strip()

OCR_DPI = int(os.getenv("OCR_DPI", "200"))

OCR_MIN_TEXT_LENGTH = int(os.getenv("OCR_MIN_TEXT_LENGTH", "40"))

OCR_MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", "6"))


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_NAME = "SYLLABUSIQ"

APP_VERSION = "1.0.0"

DEBUG = True


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_config():

    errors = []

    if not NVIDIA_API_KEY:
        errors.append(
            "NVIDIA_API_KEY is missing from .env"
        )

    if not NVIDIA_BASE_URL:
        errors.append(
            "NVIDIA_BASE_URL is missing"
        )

    if not NVIDIA_MODEL:
        errors.append(
            "NVIDIA_MODEL is missing"
        )

    if not MONGODB_URI:
        errors.append(
            "MONGODB_URI is missing"
        )

    if not MONGODB_DATABASE:
        errors.append(
            "MONGODB_DATABASE is missing"
        )

    return errors


# ============================================================
# CONFIG SUMMARY
# ============================================================

def get_config_summary():

    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "nvidia_model": NVIDIA_MODEL,
        "nvidia_base_url": NVIDIA_BASE_URL,
        "mongodb_database": MONGODB_DATABASE,
        "ocr_enabled": OCR_ENABLED,
        "ocr_language": OCR_LANGUAGE,
        "upload_directory": str(UPLOAD_DIR),
        "output_directory": str(OUTPUT_DIR),
        "nvidia_api_key_configured": bool(NVIDIA_API_KEY),
    }