"""Tests for evaluate.py — scoring logic, code extraction, number parsing."""

from evaluate import (
    evaluate,
    _normalize_text,
    _keyword_score,
    _extract_code_blocks,
    _extract_numbers,
    _clean_response_to_python,
    _split_test_blocks,
)


# --- _normalize_text ---

def test_normalize_strips_latex_delimiters():
    assert "x+1" in _normalize_text("$$x+1$$")
    assert "y" in _normalize_text("$y$")


def test_normalize_converts_frac():
    result = _normalize_text(r"\frac{a}{b}")
    assert "(a/b)" in result


def test_normalize_strips_bold():
    assert "hello" in _normalize_text("**hello**")
    assert "world" in _normalize_text("__world__")


# --- _keyword_score ---

def test_keyword_score_all_found():
    ratio, found, missing = _keyword_score(["def", "return"], "def foo():\n    return 1")
    assert ratio == 1.0
    assert len(missing) == 0


def test_keyword_score_partial():
    ratio, found, missing = _keyword_score(["def", "class", "return"], "def foo(): return 1")
    assert ratio == 2 / 3
    assert "class" in missing


def test_keyword_score_case_insensitive():
    ratio, _, _ = _keyword_score(["FizzBuzz"], "fizzbuzz is fun")
    assert ratio == 1.0


def test_keyword_score_empty_keywords():
    ratio, _, _ = _keyword_score([], "anything")
    assert ratio == 1.0


def test_keyword_score_in_normalized_text():
    # LaTeX-wrapped number should still match after normalization
    ratio, _, _ = _keyword_score(["5000"], r"The area is $5000$ square meters")
    assert ratio == 1.0


# --- _extract_code_blocks ---

def test_extract_fenced_python():
    response = "Here's the code:\n```python\ndef foo():\n    return 1\n```\nDone."
    blocks = _extract_code_blocks(response)
    assert len(blocks) == 1
    assert "def foo" in blocks[0]


def test_extract_unfenced():
    response = "Code:\n```\nx = 1\ny = 2\n```"
    blocks = _extract_code_blocks(response)
    assert len(blocks) == 1


def test_extract_no_code():
    blocks = _extract_code_blocks("Just some plain text with no code at all.")
    assert blocks == []


def test_extract_indented_block():
    response = "Here:\n    def foo():\n        return 1\n\nEnd."
    blocks = _extract_code_blocks(response)
    assert len(blocks) >= 1
    assert "def foo" in blocks[0]


# --- _extract_numbers ---

def test_extract_plain_numbers():
    nums = _extract_numbers("The answer is 42 and 3.14")
    assert 42.0 in nums
    assert 3.14 in nums


def test_extract_currency():
    nums = _extract_numbers("Total: $1,234.56")
    assert 1234.56 in nums


def test_extract_negative():
    nums = _extract_numbers("Temperature is -5 degrees")
    assert -5.0 in nums


def test_extract_from_latex():
    nums = _extract_numbers(r"Result: $\frac{1}{4} = 0.25$")
    assert 0.25 in nums


# --- _clean_response_to_python ---

def test_clean_raw_python():
    code = "def foo():\n    return 1\n"
    assert _clean_response_to_python(code) == code.strip()


def test_clean_markdown_wrapped():
    response = "```python\ndef foo():\n    return 1\n```"
    result = _clean_response_to_python(response)
    assert "def foo" in result
    assert "```" not in result


def test_clean_strips_eos_tokens():
    code = "def foo():\n    return 1\n<|im_end|>"
    result = _clean_response_to_python(code)
    assert "<|im_end|>" not in result
    assert "def foo" in result


def test_clean_strips_eos_token():
    code = "def foo():\n    return 1\n<eos>"
    result = _clean_response_to_python(code)
    assert "<eos>" not in result


def test_clean_trims_trailing_garbage():
    code = "def foo():\n    return 1\n\n# Example usage:\nfoo()\nsome broken line that won't compile {"
    result = _clean_response_to_python(code)
    # Should compile
    compile(result, "<string>", "exec")


# --- _split_test_blocks ---

def test_split_groups_indented():
    # _split_test_blocks groups indented continuation with parent,
    # but except: is not indented so it starts a new block
    code = "try:\n    x = 1/0\nexcept:\n    pass\nassert True"
    blocks = _split_test_blocks(code)
    assert len(blocks) == 3
    assert "try:" in blocks[0]
    assert "except:" in blocks[1]
    assert "assert True" in blocks[2]


def test_split_drops_standalone_print_pass():
    code = "assert 1 == 1\nprint('PASS')\nassert 2 == 2"
    blocks = _split_test_blocks(code)
    assert all("print('PASS')" not in b for b in blocks)
    assert len(blocks) == 2


# --- evaluate() dispatch ---

def test_evaluate_empty_response():
    score, notes = evaluate({"category": "coding"}, "")
    assert score == 0.0
    assert "empty" in notes


def test_evaluate_empty_whitespace():
    score, notes = evaluate({"category": "math"}, "   \n  ")
    assert score == 0.0


def test_evaluate_unknown_category():
    prompt = {"category": "unknown_cat", "reference_keywords": ["hello"]}
    score, _ = evaluate(prompt, "hello world")
    assert score > 0
