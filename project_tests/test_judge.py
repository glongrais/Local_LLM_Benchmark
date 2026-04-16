"""Tests for judge.py — _parse_judge_response edge cases."""

from judge import _parse_judge_response


def test_clean_json():
    content = '{"score": 8, "correctness": 3, "completeness": 3, "quality": 2, "reason": "good response"}'
    score, reason = _parse_judge_response(content)
    assert score == 8.0
    assert reason == "good response"


def test_json_in_markdown_fence():
    content = '```json\n{"score": 7, "reason": "decent"}\n```'
    score, reason = _parse_judge_response(content)
    assert score == 7.0
    assert reason == "decent"


def test_score_clamped_high():
    content = '{"score": 15, "reason": "amazing"}'
    score, _ = _parse_judge_response(content)
    assert score == 10.0


def test_score_clamped_low():
    content = '{"score": -3, "reason": "terrible"}'
    score, _ = _parse_judge_response(content)
    assert score == 0.0


def test_partial_json_fallback():
    content = 'Some preamble text... "score": 6, "reason": "ok" and more text'
    score, _ = _parse_judge_response(content)
    assert score == 6.0


def test_x_out_of_10_fallback():
    content = "I would rate this response 7/10 because it covers the basics."
    score, _ = _parse_judge_response(content)
    assert score == 7.0


def test_thinking_model_with_reasoning_then_json():
    content = (
        "Let me analyze this response carefully.\n"
        "The code is well structured but has a minor bug.\n"
        "The explanation is clear.\n\n"
        '{"score": 9, "correctness": 4, "completeness": 3, "quality": 2, "reason": "solid"}'
    )
    score, reason = _parse_judge_response(content)
    assert score == 9.0
    assert reason == "solid"


def test_unparseable():
    content = "This is just random text with no scores or numbers at all."
    score, reason = _parse_judge_response(content)
    assert score == 0.0
    assert "unparseable" in reason


def test_json_with_braces_in_reason():
    content = '{"score": 5, "reason": "code uses dict comprehension {k: v}"}'
    score, _ = _parse_judge_response(content)
    assert score == 5.0


def test_zero_score():
    content = '{"score": 0, "reason": "completely wrong"}'
    score, reason = _parse_judge_response(content)
    assert score == 0.0
    assert reason == "completely wrong"
