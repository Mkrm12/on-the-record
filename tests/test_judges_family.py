"""as a structural guarantee: no code path may pick an ANSWER judge from the generator's family."""

from __future__ import annotations

import re
from pathlib import Path

from app.config import settings
from goldset import judges

ROOT = Path(__file__).resolve().parents[1]
CALL = re.compile(r"\b(?:J|judges)\.(panel|primary)\(([^)]*)\)")


def test_every_judge_selection_in_the_code_excludes_the_generators_family():
    calls, offenders = 0, []
    for path in [*(ROOT / "scripts").glob("*.py"), *(ROOT / "src").rglob("*.py")]:
        for match in CALL.finditer(path.read_text(encoding="utf-8")):
            calls += 1
            if "exclude_family" not in match.group(2):
                offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    assert calls >= 4, "the scan found the judge selections it guards"
    assert offenders == []


def test_the_generator_family_is_the_generators():
    """If the generator ever moves to another family, this names the constant that must move with it."""
    assert judges.GENERATOR_FAMILY == "deepseek" and settings.bulk_model.startswith("deepseek")


def test_with_the_keys_this_machine_has_there_is_no_api_answer_judge(monkeypatch):
    """DEEPSEEK_KEY only: the cross-family API judge is absent, and the evaluation must say so and point at the offline
    Gemini judge rather than fall back to DeepSeek.
    """
    monkeypatch.setenv("DEEPSEEK_KEY", "x")
    for name in (*judges.KEY_NAMES["anthropic"], *judges.KEY_NAMES["google"]):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(judges, "_DOTENV_CACHE", {})
    available = judges.panel(exclude_family=judges.GENERATOR_FAMILY)
    assert judges.primary(available, exclude_family=judges.GENERATOR_FAMILY) is None
