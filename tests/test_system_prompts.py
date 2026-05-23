from __future__ import annotations

from pathlib import Path

from cagent.system_prompts import (
    OPERATING_CONTRACT,
    PLAN_SYSTEM_PROMPT,
    implementation_system_prompt,
)


def test_implementation_prompt_includes_shared_operating_contract() -> None:
    prompt = implementation_system_prompt(Path("/workspace"))

    assert OPERATING_CONTRACT in prompt
    assert "Current directory: /workspace" in prompt


def test_plan_prompt_includes_planning_contract() -> None:
    assert OPERATING_CONTRACT in PLAN_SYSTEM_PROMPT
    assert "planning mode" in PLAN_SYSTEM_PROMPT
    assert "During planning, do not perform writes, deletes, or shell execution." in (
        PLAN_SYSTEM_PROMPT
    )
