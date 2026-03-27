import pytest

pytest.importorskip("langchain_core")

from app.utils.prompting import build_rag_prompt, build_refine_rag_prompt


def test_prompt_contains_guardrails() -> None:
    prompt = build_rag_prompt(question="What is this?", context="Some context")

    assert "Answer only with information from the provided context" in prompt
    assert "I don't know based on the provided documents." in prompt
    assert "Some context" in prompt


def test_refine_prompt_contains_grounding_rules() -> None:
    prompt = build_refine_rag_prompt(
        question="What is this?",
        context="Some context",
        draft_answer="Draft answer",
    )

    assert "Use only the provided context." in prompt
    assert "I don't know based on the provided documents." in prompt
    assert "Some context" in prompt
    assert "Draft answer" in prompt
