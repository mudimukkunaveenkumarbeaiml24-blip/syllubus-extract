import os
import logging
import threading
import time
import httpx

logger = logging.getLogger("syllabusiq.nvidia")

try:
    from app.config import (
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        NVIDIA_MODEL,
        LLM_TEMPERATURE,
        LLM_TOP_P,
        LLM_MAX_TOKENS,
        LLM_TIMEOUT,
    )
except ImportError:
    from config import (
        NVIDIA_API_KEY,
        NVIDIA_BASE_URL,
        NVIDIA_MODEL,
        LLM_TEMPERATURE,
        LLM_TOP_P,
        LLM_MAX_TOKENS,
        LLM_TIMEOUT,
    )

LLM_REQUEST_LOCK = threading.Lock()

# Verified responsive fallback models on NVIDIA API
DEFAULT_FALLBACK_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "meta/muse-glimmer-30b",
    "poolside/laguna-xs-2.1",
    "mistralai/mistral-nemotron",
]


class RateLimitException(Exception):
    """Raised when NVIDIA rate limit is hit and all backoff attempts are exhausted."""
    def __init__(self, message="NVIDIA API rate limit reached. Please wait and retry.", retry_after=10):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.error_code = "NVIDIA_RATE_LIMIT"


def generate_text(
    prompt: str,
    system_prompt: str = "",
    temperature: float = None,
    max_tokens: int = None,
    timeout_seconds: float = 30.0,
):
    """
    Issues single-concurrency LLM requests with automatic Multi-Tier failover across models & keys.
    Uses httpx directly to avoid binary dependency/DLL issues.
    """
    with LLM_REQUEST_LOCK:
        primary_key = os.getenv("NVIDIA_API_KEY", "").strip() or NVIDIA_API_KEY
        primary_model = os.getenv("NVIDIA_MODEL", "").strip() or NVIDIA_MODEL or "meta/llama-3.2-11b-vision-instruct"

        backup_key = os.getenv("NVIDIA_BACKUP_API_KEY", "").strip() or primary_key
        backup_model = os.getenv("NVIDIA_BACKUP_MODEL", "").strip() or "meta/muse-glimmer-30b"

        candidate_tiers = []
        if primary_key:
            candidate_tiers.append({"name": "Primary", "model": primary_model, "key": primary_key})
        if backup_key and backup_model != primary_model:
            candidate_tiers.append({"name": "Backup", "model": backup_model, "key": backup_key})

        # Add additional safety fallback models
        for fb_model in DEFAULT_FALLBACK_MODELS:
            if fb_model != primary_model and fb_model != backup_model:
                candidate_tiers.append({"name": f"SafetyFallback-{fb_model.split('/')[-1]}", "model": fb_model, "key": primary_key or backup_key})

        if not candidate_tiers:
            raise RuntimeError("No valid NVIDIA API keys configured in .env.")

        last_exception = None
        url = f"{NVIDIA_BASE_URL.rstrip('/')}/chat/completions"

        for tier_idx, tier in enumerate(candidate_tiers, 1):
            tier_name = tier["name"]
            active_model = tier["model"]
            active_key = tier["key"]

            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": active_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            system_prompt
                            or "You are a precise document extraction assistant."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": (
                    LLM_TEMPERATURE
                    if temperature is None
                    else temperature
                ),
                "top_p": LLM_TOP_P,
                "max_tokens": (
                    LLM_MAX_TOKENS
                    if max_tokens is None
                    else max_tokens
                ),
                "stream": False,
            }

            logger.info(f"[LLM] Attempting {tier_name} model '{active_model}'...")

            # 2 attempts per tier
            for attempt in range(2):
                try:
                    with httpx.Client(
                        timeout=httpx.Timeout(timeout_seconds, connect=10.0, read=timeout_seconds, write=15.0)
                    ) as client:
                        response = client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                        res_data = response.json()

                    choices = res_data.get("choices", [])
                    if not choices:
                        raise RuntimeError(f"{active_model} returned no choices.")

                    message = choices[0].get("message", {})
                    content = message.get("content")

                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("text"):
                                text_parts.append(item.get("text"))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        content = "".join(text_parts)

                    if content and str(content).strip():
                        return str(content).strip()

                    reasoning = message.get("reasoning_content")
                    if reasoning and str(reasoning).strip():
                        return str(reasoning).strip()

                    raise RuntimeError(f"{active_model} returned empty content.")

                except Exception as e:
                    last_exception = e
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "429" in err_str
                        or "rate limit" in err_str
                        or "too many requests" in err_str
                        or "quota" in err_str
                    )

                    if is_rate_limit and attempt == 0:
                        logger.warning(
                            f"[{tier_name} 429 RATE LIMIT] Rate limited on '{active_model}'. "
                            f"Waiting 2s before retry..."
                        )
                        time.sleep(2)
                        continue
                    else:
                        logger.warning(
                            f"[{tier_name} FAILED] {active_model} error ({e}). "
                            + (f"Switching to next candidate ({tier_idx + 1}/{len(candidate_tiers)})..." if tier_idx < len(candidate_tiers) else "No more candidates.")
                        )
                        break

        # If all tiers exhausted
        logger.error("[LLM FAILOVER] All model tiers exhausted.")
        if last_exception and ("429" in str(last_exception).lower() or "rate limit" in str(last_exception).lower()):
            raise RateLimitException(
                message="All NVIDIA model endpoints hit rate limits. Please wait and retry.",
                retry_after=10
            ) from last_exception
        raise last_exception or RuntimeError("All LLM tiers failed.")


def test_nvidia():
    return generate_text(
        prompt="Hello, return JSON: {\"status\": \"connected\", \"model\": \"active\"}",
        system_prompt="You are a helpful JSON extraction assistant. Return valid JSON only.",
        temperature=0.1,
        max_tokens=256,
        timeout_seconds=15.0,
    )