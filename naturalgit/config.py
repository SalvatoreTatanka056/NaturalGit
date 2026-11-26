"""
Configuration management for NaturalGit.
Stored in ~/.naturalgit/config.json
"""
import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path.home() / ".naturalgit"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "auto",           # auto, ollama, gemini, openai, heuristic, local
    "gemini_api_key": "",
    "gemini_model": "gemini-flash-lite-latest",
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "openai_model": "gpt-4o-mini",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3",
    "local_model_path": "",       # Path to .asmbin model file for llmasm
    "local_backend": "auto",      # auto, c, numpy
    "local_threads": 4,           # Number of CPU worker threads
    "auto_confirm": False,
    "language": "it"
}


def load_config() -> Dict[str, Any]:
    """Loads configuration merging defaults, saved file, and environment variables."""
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg.update(saved)
        except Exception:
            pass

    # Environment variable overrides
    if os.getenv("GEMINI_API_KEY"):
        cfg["gemini_api_key"] = os.getenv("GEMINI_API_KEY")
    if os.getenv("OPENAI_API_KEY"):
        cfg["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_BASE_URL"):
        cfg["openai_base_url"] = os.getenv("OPENAI_BASE_URL")
    if os.getenv("OLLAMA_HOST"):
        cfg["ollama_url"] = os.getenv("OLLAMA_HOST")
    if os.getenv("NGIT_LOCAL_MODEL"):
        cfg["local_model_path"] = os.getenv("NGIT_LOCAL_MODEL")

    return cfg


def save_config(cfg: Dict[str, Any]) -> bool:
    """Saves configuration to ~/.naturalgit/config.json."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False
