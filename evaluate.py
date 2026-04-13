"""Hybrid auto-evaluator for benchmark responses.

Scoring strategy per category:
- coding:    keyword matching + extract Python code and run it
- math:      check for correct final answer + intermediate steps (handles LaTeX)
- reasoning: keyword matching + verify logical conclusions
- general:   keyword matching + structural/instruction checks
"""

from __future__ import annotations

import re
import subprocess
import sys


def evaluate(prompt: dict, response: str) -> tuple[float, str]:
    """Score a response 0-10. Returns (score, explanation)."""
    if not response or not response.strip():
        return 0.0, "empty response"

    category = prompt.get("category", "")
    evaluators = {
        "coding": _eval_coding,
        "math": _eval_math,
        "reasoning": _eval_reasoning,
        "general": _eval_general,
        "agentic_coding": _eval_agentic_coding,
        "executable": _eval_executable,
        "ml": _eval_executable,
    }
    eval_fn = evaluators.get(category, _eval_keywords_only)
    return eval_fn(prompt, response)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Strip LaTeX, markdown formatting, and normalize whitespace for matching."""
    t = text
    # Strip LaTeX math delimiters: $$...$$ and $...$
    t = re.sub(r"\$\$.*?\$\$", lambda m: m.group().replace("$$", ""), t, flags=re.DOTALL)
    t = re.sub(r"\$([^$]+?)\$", r"\1", t)
    # Strip LaTeX commands like \text{...}, \frac{a}{b}, \times, etc.
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1/\2)", t)
    t = re.sub(r"\\(?:times|cdot|div)", " ", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    # Strip bold/italic markdown: **text**, *text*, __text__
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    return t


def _keyword_score(keywords: list[str], response: str) -> tuple[float, list[str], list[str]]:
    """Check what fraction of reference keywords appear in the response.
    Searches both raw and normalized text.
    """
    if not keywords:
        return 1.0, [], []
    raw_lower = response.lower()
    norm_lower = _normalize_text(response).lower()
    found = []
    missing = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in raw_lower or kw_lower in norm_lower:
            found.append(kw)
        else:
            missing.append(kw)
    ratio = len(found) / len(keywords)
    return ratio, found, missing


def _extract_code_blocks(response: str) -> list[str]:
    """Extract Python code blocks from markdown-formatted response."""
    # Try fenced code blocks first (```python or ```)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks

    # Try unfenced blocks: any ``` ... ``` pair
    blocks = re.findall(r"```\s*\n?(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks

    # Try indented code blocks (4 spaces or tab)
    lines = response.split("\n")
    current_block: list[str] = []
    result = []
    for line in lines:
        if line.startswith("    ") or line.startswith("\t"):
            current_block.append(line)
        elif current_block and line.strip() == "":
            current_block.append(line)
        elif current_block:
            # Only keep blocks that look like Python
            block = "\n".join(current_block)
            if re.search(r"(def |class |import |for |if |while |return )", block):
                result.append(block)
            current_block = []
    if current_block:
        block = "\n".join(current_block)
        if re.search(r"(def |class |import |for |if |while |return )", block):
            result.append(block)

    # Last resort: find lines that look like Python code (def, class, import blocks)
    if not result:
        code_lines: list[str] = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"(def |class |import |from .* import |@)", stripped):
                in_code = True
            if in_code:
                # Stop at clearly non-code lines (markdown headers, plain sentences)
                if stripped and not stripped.startswith("#") and not re.match(r"^[A-Z][a-z].*[.!?]$", stripped):
                    code_lines.append(line)
                elif not stripped:
                    code_lines.append(line)
                else:
                    if code_lines:
                        result.append("\n".join(code_lines))
                        code_lines = []
                    in_code = False
        if code_lines:
            result.append("\n".join(code_lines))

    return result


def _run_code(code: str, test_code: str = "", timeout: int = 10) -> tuple[bool, str]:
    """Run Python code in a subprocess. Returns (success, output/error)."""
    full_code = code + "\n" + test_code
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()[-200:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text, handling LaTeX and currency formatting."""
    normalized = _normalize_text(text)
    # Remove currency symbols and commas in numbers
    normalized = re.sub(r"[$€£]", "", normalized)
    normalized = re.sub(r"(\d),(\d{3})", r"\1\2", normalized)
    matches = re.findall(r"-?\d+\.?\d*", normalized)
    result = []
    for m in matches:
        try:
            result.append(float(m))
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# Category evaluators
# ---------------------------------------------------------------------------

def _eval_coding(prompt: dict, response: str) -> tuple[float, str]:
    name = prompt.get("name", "")
    keywords = prompt.get("reference_keywords", [])
    kw_ratio, found, missing = _keyword_score(keywords, response)

    notes = []
    code_score = 0.0
    blocks = _extract_code_blocks(response)

    if not blocks:
        notes.append("no code blocks found")
        if missing:
            notes.append(f"missing keywords: {missing}")
        return round(kw_ratio * 5, 1), "; ".join(notes)

    notes.append(f"{len(blocks)} code block(s) found")

    test_cases = {
        "fizzbuzz": (
            "result = fizzbuzz(15)\n"
            "assert result[2] == 'Fizz', f'Expected Fizz got {result[2]}'\n"
            "assert result[4] == 'Buzz', f'Expected Buzz got {result[4]}'\n"
            "assert result[14] == 'FizzBuzz', f'Expected FizzBuzz got {result[14]}'\n"
            "assert result[0] == '1', f'Expected 1 got {result[0]}'\n"
            "assert len(result) == 15\n"
            "print('PASS')\n"
        ),
        "binary_search": (
            "assert binary_search([1,2,3,4,5], 3) == 2\n"
            "assert binary_search([1,2,3,4,5], 6) == -1\n"
            "assert binary_search([], 1) == -1\n"
            "assert binary_search([1], 1) == 0\n"
            "print('PASS')\n"
        ),
    }

    test_code = test_cases.get(name, "print('PASS')\n")

    # Try each block (and combinations for multi-block responses)
    all_code = "\n\n".join(blocks)
    attempts = blocks + ([all_code] if len(blocks) > 1 else [])

    for block in attempts:
        success, output = _run_code(block, test_code)
        if success and "PASS" in output:
            code_score = 1.0
            notes.append("code passes tests")
            break
        elif success:
            code_score = max(code_score, 0.6)
            notes.append("code runs but tests unclear")
        else:
            if code_score < 0.3:
                # At least it has code structure
                code_score = 0.3
                notes.append(f"code error: {output[:80]}")

    if missing:
        notes.append(f"missing keywords: {missing}")

    # 50% keywords, 50% code execution
    score = (kw_ratio * 5) + (code_score * 5)
    return round(score, 1), "; ".join(notes)


def _eval_math(prompt: dict, response: str) -> tuple[float, str]:
    name = prompt.get("name", "")
    keywords = prompt.get("reference_keywords", [])

    expected_answers = {
        "multi_step_arithmetic": {
            "final": [31.645, 31.65],
            "intermediates": [63.0, 5.355, 68.355],
            "tolerance": 0.05,
        },
        "algebra_word_problem": {
            "final": [1.857, 1.86, 13/7],
            "intermediates": [120, 260],
            "tolerance": 0.15,
        },
        "probability": {
            "final": [0.25, 1/4],
            "alt_forms": ["1/4", "25%", "30/120"],
            "intermediates": [30, 120],
            "tolerance": 0.02,
        },
        "optimization": {
            "final": [5000],
            "intermediates": [100, 50],
            "tolerance": 1.0,
        },
    }

    notes = []
    answer_score = 0.0
    step_score = 0.0

    response_numbers = _extract_numbers(response)
    normalized = _normalize_text(response).lower()

    if name in expected_answers:
        spec = expected_answers[name]
        tol = spec["tolerance"]

        # Check final answer (numeric)
        finals = spec["final"]
        for f in finals:
            if any(abs(n - f) < tol for n in response_numbers):
                answer_score = 1.0
                notes.append(f"correct final answer ({f})")
                break

        # Also check alternative text forms (e.g. "1/4", "25%")
        if answer_score == 0:
            for alt in spec.get("alt_forms", []):
                if alt.lower() in normalized:
                    answer_score = 1.0
                    notes.append(f"correct final answer ({alt})")
                    break

        if answer_score == 0:
            notes.append(f"final answer not found (expected ~{finals[0]})")

        # Check intermediate steps
        intermediates = spec.get("intermediates", [])
        if intermediates:
            found_steps = sum(
                1 for s in intermediates
                if any(abs(n - s) < max(tol, 0.5) for n in response_numbers)
            )
            step_score = found_steps / len(intermediates)
            if found_steps == len(intermediates):
                notes.append("all intermediate steps shown")
            else:
                notes.append(f"intermediate steps: {found_steps}/{len(intermediates)}")
    else:
        kw_ratio, found, missing = _keyword_score(keywords, response)
        answer_score = kw_ratio
        if missing:
            notes.append(f"missing keywords: {missing}")

    # 60% final answer, 40% showing work
    score = (answer_score * 6) + (step_score * 4)
    return round(score, 1), "; ".join(notes)


def _eval_reasoning(prompt: dict, response: str) -> tuple[float, str]:
    name = prompt.get("name", "")
    keywords = prompt.get("reference_keywords", [])
    kw_ratio, found, missing = _keyword_score(keywords, response)
    normalized = _normalize_text(response)

    notes = []
    correctness = 0.0

    conclusion_checks = {
        "river_crossing": {
            "required_patterns": [
                r"goat",  # must mention the goat (central to the puzzle)
                r"(?:take|bring|carry|cross).*(?:goat|wolf|cabbage)|(?:goat|wolf|cabbage).*(?:take|bring|carry|cross)",
            ],
            "min_steps": 5,
        },
        "knights_knaves": {
            "required_patterns": [
                r"A.*knight|A\s*[:=]\s*knight|A\s+is\s+(?:a\s+)?knight",
                r"B.*knave|B\s*[:=]\s*knave|B\s+is\s+(?:a\s+)?knave",
                r"C.*knight|C\s*[:=]\s*knight|C\s+is\s+(?:a\s+)?knight",
            ],
        },
        "lateral_thinking": {
            "required_patterns": [
                r"[Mm]onopoly|board\s*game|game\s+(?:of\s+)?[Mm]onopoly|game.*piece|token|playing.*game",
            ],
        },
        "causal_reasoning": {
            "required_patterns": [
                r"correlat|causation|caus(?:e|al)|does\s+not\s+(?:necessarily\s+)?(?:mean|imply|prove)",
                r"confound|third.variable|lurking|hidden|alternative|other\s+(?:factor|explanation|reason)",
                r"selection|self.select|bias|health(?:ier|y)\s+(?:employee|worker|people)",
            ],
            "min_explanations": 3,
        },
    }

    if name in conclusion_checks:
        checks = conclusion_checks[name]
        patterns = checks.get("required_patterns", [])
        matched = sum(1 for p in patterns if re.search(p, response, re.IGNORECASE))
        correctness = matched / len(patterns) if patterns else 0

        if matched == len(patterns):
            notes.append("correct conclusion")
        else:
            notes.append(f"conclusion check: {matched}/{len(patterns)} patterns matched")

        min_steps = checks.get("min_steps")
        if min_steps:
            # Count steps more broadly: numbered items, "step" mentions, trip descriptions
            step_markers = re.findall(
                r"(?:step\s*\d|trip\s*\d|\d+[\.\):]|\*\*\d|\-\s+\w|#{1,3}\s+\w)",
                response, re.IGNORECASE
            )
            if len(step_markers) >= min_steps:
                notes.append(f"sufficient detail ({len(step_markers)} steps)")
            else:
                correctness *= 0.8
                notes.append(f"may lack detail ({len(step_markers)} step markers)")

        min_expl = checks.get("min_explanations")
        if min_expl:
            explanations = re.findall(r"(?:^\s*[\d\-\*•►]|\d+[\.\)])", response, re.MULTILINE)
            if len(explanations) >= min_expl:
                notes.append(f"{len(explanations)} explanations provided")
    else:
        correctness = kw_ratio

    if missing:
        notes.append(f"missing keywords: {missing}")

    score = (correctness * 6) + (kw_ratio * 4)
    return round(min(score, 10.0), 1), "; ".join(notes)


def _eval_general(prompt: dict, response: str) -> tuple[float, str]:
    name = prompt.get("name", "")
    keywords = prompt.get("reference_keywords", [])
    kw_ratio, found, missing = _keyword_score(keywords, response)

    notes = []
    structure_score = 0.0

    if name == "summarization":
        words = len(response.split())
        if words <= 250:
            structure_score = 1.0
            notes.append(f"{words} words (within limit)")
        elif words <= 350:
            structure_score = 0.5
            notes.append(f"{words} words (slightly over)")
        else:
            structure_score = 0.2
            notes.append(f"{words} words (way over 200 limit)")

    elif name == "comparison":
        aspects = ["architect", "efficien", "learning curve", "tool", "when to"]
        covered = sum(1 for a in aspects if a.lower() in response.lower())
        structure_score = covered / len(aspects)
        notes.append(f"covers {covered}/{len(aspects)} requested aspects")

    elif name == "instruction_following":
        # Split more carefully: by line for structured responses, by sentence otherwise
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        # Filter: keep lines that look like actual sentences (>15 chars, not just a header)
        sentences = [l for l in lines if len(l) > 15 and not l.startswith("#")]
        # Also try sentence splitting if line-based didn't work well
        if len(sentences) < 3:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if len(s.strip()) > 15]

        score_parts = []

        # Check count
        if len(sentences) >= 5:
            score_parts.append(1.0)
            notes.append(f"{len(sentences)} sentences")
        else:
            score_parts.append(len(sentences) / 5)
            notes.append(f"only {len(sentences)} sentences (need 5)")

        # Check LEARN ordering — strip markdown bold markers
        expected_letters = list("LEARN")
        letter_matches = 0
        for i, letter in enumerate(expected_letters):
            if i < len(sentences):
                # Strip leading markdown bold: **L**, *L*, __L__
                clean = re.sub(r"^[\*_]{1,2}([A-Za-z])[\*_]{0,2}", r"\1", sentences[i])
                if clean.upper().startswith(letter):
                    letter_matches += 1
        score_parts.append(letter_matches / 5)
        if letter_matches == 5:
            notes.append("LEARN ordering correct")
        else:
            notes.append(f"LEARN start: {letter_matches}/5")

        # Check numbers in each sentence
        sentences_with_numbers = sum(1 for s in sentences[:5] if re.search(r'\d+', s))
        score_parts.append(sentences_with_numbers / 5)
        if sentences_with_numbers < 5:
            notes.append(f"numbers present: {sentences_with_numbers}/5")

        structure_score = sum(score_parts) / len(score_parts)
    else:
        structure_score = kw_ratio

    if missing:
        notes.append(f"missing keywords: {missing}")

    score = (kw_ratio * 5) + (structure_score * 5)
    return round(min(score, 10.0), 1), "; ".join(notes)


def _eval_agentic_coding(prompt: dict, response: str) -> tuple[float, str]:
    """Evaluate complex, multi-faceted coding tasks.

    Scoring (out of 10):
      - 3 pts: keyword coverage (are the right concepts mentioned?)
      - 3 pts: code substance (enough code blocks, functions/classes defined)
      - 2 pts: architectural quality (separation of concerns, classes, error handling)
      - 2 pts: prompt-specific structural checks
    """
    name = prompt.get("name", "")
    keywords = prompt.get("reference_keywords", [])
    kw_ratio, found, missing = _keyword_score(keywords, response)

    notes = []
    code_blocks = _extract_code_blocks(response)
    all_code = "\n".join(code_blocks)
    response_lower = response.lower()

    # --- 1. Keyword score (3 pts) ---
    kw_score = kw_ratio * 3
    if missing:
        notes.append(f"missing keywords: {missing}")

    # --- 2. Code substance (3 pts) ---
    substance = 0.0
    code_lines = sum(len(b.strip().split("\n")) for b in code_blocks)
    func_count = len(re.findall(r"^\s*(?:def|async def) \w+", all_code, re.MULTILINE))
    class_count = len(re.findall(r"^\s*class \w+", all_code, re.MULTILINE))

    if code_lines >= 100:
        substance += 1.0
    elif code_lines >= 50:
        substance += 0.6
    elif code_lines >= 20:
        substance += 0.3
    notes.append(f"{code_lines} lines of code")

    if func_count >= 5:
        substance += 1.0
    elif func_count >= 3:
        substance += 0.6
    elif func_count >= 1:
        substance += 0.3
    notes.append(f"{func_count} functions, {class_count} classes")

    if class_count >= 2:
        substance += 1.0
    elif class_count >= 1:
        substance += 0.6
    elif func_count >= 5:
        substance += 0.4  # functional style is ok too
    substance = min(substance, 3.0)

    # --- 3. Architectural quality (2 pts) ---
    arch = 0.0
    # Error handling
    if re.search(r"try\s*:|except\s+\w+|raise\s+\w+", all_code):
        arch += 0.5
    # Type hints
    if re.search(r"def \w+\(.*:\s*\w+", all_code):
        arch += 0.3
    # Docstrings or comments showing reasoning
    if re.search(r'""".*?"""|\'\'\'.*?\'\'\'', all_code, re.DOTALL) or response.count("# ") >= 3:
        arch += 0.2
    # Dependency injection / no globals
    if re.search(r"def __init__\(self.*\w+", all_code):
        arch += 0.5
    # Separation of concerns (multiple classes or clear module structure)
    if class_count >= 2 or (func_count >= 8 and "import" in all_code):
        arch += 0.5
    arch = min(arch, 2.0)

    # --- 4. Prompt-specific checks (2 pts) ---
    specific = 0.0

    if name == "project_from_spec":
        checks = [
            (r"argparse|ArgumentParser", "CLI with argparse"),
            (r"yaml\.safe_load|yaml\.load", "YAML parsing"),
            (r"ThreadPoolExecutor|asyncio", "parallel execution"),
            (r"topological|topo_sort|in_degree|kahn|dfs.*visit", "topological sort"),
            (r"cycle|circular|cycl", "cycle detection"),
            (r"dry.?run|--dry", "dry-run support"),
            (r"timeout|TimeoutExpired", "timeout handling"),
            (r"subprocess|Popen|run\(", "subprocess execution"),
        ]
        matched = sum(1 for pat, _ in checks if re.search(pat, all_code + response, re.IGNORECASE))
        specific = (matched / len(checks)) * 2
        notes.append(f"spec coverage: {matched}/{len(checks)} features")

    elif name == "refactor_with_context":
        checks = [
            (r"class\s+UserRepository|class\s+.*Repository", "UserRepository class"),
            (r"class\s+AuthService|class\s+.*Service", "AuthService class"),
            (r"class\s+SessionManager|class\s+.*Session", "SessionManager class"),
            (r"bcrypt|argon2|scrypt|pbkdf2", "proper password hashing"),
            (r"def test_|unittest|pytest|mock|Mock", "test code"),
            (r"inject|__init__\(self,\s*\w+", "dependency injection"),
        ]
        matched = sum(1 for pat, _ in checks if re.search(pat, all_code + response, re.IGNORECASE))
        specific = (matched / len(checks)) * 2
        notes.append(f"refactor coverage: {matched}/{len(checks)} goals")

    elif name == "debug_complex_system":
        # Check that key bugs are identified
        bugs = [
            (r"session.*close|close.*session|__aexit__|async with.*Session", "session not closed"),
            (r"retry.*recursi|recursion.*retry|stack|recursive", "recursive retry issue"),
            (r"process_results.*never.*clear|results.*grow|never.*remov|memory.*leak", "results list grows forever"),
            (r"cancel.*process|process.*cancel|while True.*never", "processor task issues"),
            (r"semaphore.*retry|retry.*outside.*semaphore|deadlock", "semaphore held during retry"),
        ]
        matched = sum(1 for pat, _ in bugs if re.search(pat, response, re.IGNORECASE))
        specific = (matched / len(bugs)) * 2
        notes.append(f"bugs found: {matched}/{len(bugs)}")

    elif name == "architecture_decision":
        # Check each decision area is addressed with recommendation + tradeoff
        areas = [
            (r"(?:tauri|electron|native).*(?:recommend|pick|choose|suggest)|recommendation.*(?:tauri|electron)", "desktop framework"),
            (r"(?:fts5?|tantivy|lunr|search).*(?:recommend|pick)|recommendation.*(?:fts|search|sqlite)", "search engine"),
            (r"(?:graph|link|adjacen).*(?:recommend|pick)|recommendation.*(?:graph|sqlite|memory)", "link graph"),
            (r"(?:crdt|sync|conflict|merge).*(?:recommend|pick)|recommendation.*(?:crdt|sync|git)", "sync architecture"),
            (r"(?:plugin|wasm|lua|scripting).*(?:recommend|pick)|recommendation.*(?:plugin|wasm|script)", "plugin system"),
        ]
        matched = sum(1 for pat, _ in areas if re.search(pat, response, re.IGNORECASE))
        # Also check for tradeoff discussion
        tradeoff_count = len(re.findall(r"tradeoff|trade.off|giving up|downside|drawback|limitation", response, re.IGNORECASE))
        specific = (matched / len(areas)) * 1.5 + min(tradeoff_count / 5, 0.5)
        notes.append(f"decisions addressed: {matched}/{len(areas)}, {tradeoff_count} tradeoff mentions")

    specific = min(specific, 2.0)

    total = kw_score + substance + arch + specific
    return round(min(total, 10.0), 1), "; ".join(notes)


def _split_test_blocks(test_code: str) -> list[str]:
    """Split test code into individual runnable test blocks.

    Groups indented lines (try/except, multi-line asserts) with their parent.
    Drops standalone print('PASS').
    """
    lines = test_code.strip().split("\n")
    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "print('PASS')":
            continue

        # Indented line = continuation of current block
        if line[0] in (" ", "\t") and current:
            current.append(line)
        else:
            if current:
                blocks.append("\n".join(current))
            current = [line]

    if current:
        blocks.append("\n".join(current))
    return blocks


def _run_test_file(solution_code: str, test_file: str, timeout: int = 30) -> tuple[int, int, str]:
    """Write solution_code as solution.py in a temp dir, run test_file, parse RESULT line.

    Returns (passed, total, output).
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    tmpdir = tempfile.mkdtemp(prefix="bench_")
    try:
        # Write the model's code as solution.py
        solution_path = os.path.join(tmpdir, "solution.py")
        with open(solution_path, "w") as f:
            f.write(solution_code)

        # Resolve test file path relative to this script's directory (project root)
        test_path = Path(test_file)
        if not test_path.is_absolute():
            test_path = Path(__file__).parent / test_file
        if not test_path.exists():
            return 0, 0, f"test file not found: {test_file}"

        env = os.environ.copy()
        env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, str(test_path.resolve())],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
            env=env,
        )
        output = result.stdout + result.stderr

        # Parse RESULT line: "RESULT 3/5"
        match = re.search(r"RESULT (\d+)/(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2)), output.strip()

        # If no RESULT line but process succeeded, assume all passed
        if result.returncode == 0:
            return 0, 0, output.strip()

        return 0, 0, output.strip()[-300:]
    except subprocess.TimeoutExpired:
        return 0, 0, "timeout"
    except Exception as e:
        return 0, 0, str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _clean_response_to_python(response: str) -> str:
    """Extract Python code from a response.

    Handles: raw Python, markdown-wrapped code, trailing EOS tokens,
    and models that add example usage after the implementation.
    """
    stripped = response.strip()

    # Strip common EOS/special tokens that models append
    stripped = re.sub(r"<\|?(?:eos|end|im_end|eot_id)\|?>.*", "", stripped, flags=re.DOTALL).rstrip()

    # If it starts with markdown code fence, extract the content
    if stripped.startswith("```"):
        blocks = _extract_code_blocks(stripped)
        if blocks:
            return "\n\n".join(blocks)

    # Check if the raw response compiles as Python
    try:
        compile(stripped, "<string>", "exec")
        return stripped
    except SyntaxError:
        pass

    # Try stripping trailing lines one at a time until it compiles
    # (handles models that append example usage, comments, or broken lines)
    lines = stripped.split("\n")
    for trim in range(1, min(20, len(lines))):
        candidate = "\n".join(lines[:-trim])
        try:
            compile(candidate, "<string>", "exec")
            return candidate
        except SyntaxError:
            continue

    # Fall back to extracting code blocks
    blocks = _extract_code_blocks(stripped)
    if blocks:
        return "\n\n".join(blocks)

    # Last resort: return as-is and let it fail at runtime
    return stripped


def _eval_executable(prompt: dict, response: str) -> tuple[float, str]:
    """Evaluate by writing the model's response as solution.py and running a test file.

    If `test_file` is set, uses file-based evaluation (writes solution.py, runs test).
    Falls back to legacy inline `test_code` if no test_file.
    """
    test_file = prompt.get("test_file", "")
    num_tests = prompt.get("num_tests", 1)
    timeout = prompt.get("eval_timeout", 30)

    if test_file:
        # New file-based approach
        solution_code = _clean_response_to_python(response)
        if not solution_code.strip():
            return 0.0, "empty response (no code)"

        passed, total, output = _run_test_file(solution_code, test_file, timeout)

        if total == 0 and passed == 0:
            # Parse failures from output
            return 0.0, f"test error: {output[:150]}"

        score = (passed / num_tests) * 10

        # Collect PASS/FAIL lines for notes
        lines = output.split("\n")
        fails = [l for l in lines if l.startswith("FAIL")]
        notes = f"tests: {passed}/{num_tests} passed"
        if fails:
            notes += f"; {fails[0][:80]}"
            if len(fails) > 1:
                notes += f"; +{len(fails) - 1} more"

        return round(score, 1), notes

    # Legacy: inline test_code (for backward compatibility)
    test_code = prompt.get("test_code", "")
    if not test_code:
        return _eval_keywords_only(prompt, response)

    solution_code = _clean_response_to_python(response)
    if not solution_code.strip():
        return 0.0, "empty response (no code)"

    ok, err = _run_code(solution_code, test_code, timeout=timeout)
    if ok:
        return 10.0, f"tests: {num_tests}/{num_tests} passed (all pass)"

    return 0.0, f"tests failed: {err[:150]}"


def _eval_keywords_only(prompt: dict, response: str) -> tuple[float, str]:
    """Fallback: score based on keywords only."""
    keywords = prompt.get("reference_keywords", [])
    kw_ratio, found, missing = _keyword_score(keywords, response)
    notes = []
    if missing:
        notes.append(f"missing keywords: {missing}")
    return round(kw_ratio * 10, 1), "; ".join(notes) or "keyword match"
