"""
config.py — LEVEL_CONFIG table and path constants.
All level-varying parameters live here. Do not hardcode these elsewhere.
"""

from pathlib import Path

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "data"
SUBMISSIONS_DIR = ROOT / "submissions"
PROMPTS_DIR = ROOT / "prompts"

LEVEL_CONFIG: dict[int, dict] = {
    1: {
        "budget": 8.0,
        "gray_low": 0.30,
        "gray_high": 0.70,
        "llm_model": "gpt-4o-mini",
        "critic": False,
        "throttle_at": 0.90,   # fraction of budget where gray zone narrows
    },
    2: {
        "budget": 12.0,
        "gray_low": 0.30,
        "gray_high": 0.70,
        "llm_model": "gpt-4o-mini",
        "critic": False,
        "throttle_at": 0.90,
    },
    3: {
        "budget": 15.0,
        "gray_low": 0.25,
        "gray_high": 0.75,
        "llm_model": "gpt-4o-mini",
        "critic": False,
        "throttle_at": 0.90,
    },
    4: {
        "budget": 55.0,
        "gray_low": 0.15,
        "gray_high": 0.85,
        "llm_model": "claude-haiku-4-5-20251001",
        "critic": True,
        "throttle_at": 0.90,
    },
    5: {
        "budget": 60.0,
        "gray_low": 0.10,
        "gray_high": 0.90,
        "llm_model": "claude-sonnet-4-6",
        "critic": True,
        "throttle_at": 0.90,
    },
}
