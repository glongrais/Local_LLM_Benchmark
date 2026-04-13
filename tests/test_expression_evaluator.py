"""Test harness for expression_evaluator — imports eval_expr from solution.py"""
from solution import eval_expr

passed = 0
total = 12

def check(condition, name, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name} ({detail})")

check(eval_expr("2 + 3") == 5.0, "add")
check(eval_expr("2 + 3 * 4") == 14.0, "precedence", f"got {eval_expr('2 + 3 * 4')}")
check(eval_expr("(2 + 3) * 4") == 20.0, "parens")
check(abs(eval_expr("10 / 3") - 3.333333) < 0.001, "division")
check(eval_expr("2 * (3 + 4) - 1") == 13.0, "complex")
check(eval_expr("-5 + 3") == -2.0, "negative", f"got {eval_expr('-5 + 3')}")
check(eval_expr("(-5 + 3) * 2") == -4.0, "neg_parens")
check(eval_expr("3.5 * 2") == 7.0, "float")
check(eval_expr("((2 + 3) * (4 - 1))") == 15.0, "nested_parens")
check(eval_expr("100") == 100.0, "single_number")

try:
    eval_expr("")
    check(False, "empty_raises")
except (ValueError, Exception):
    check(True, "empty_raises")

# Bonus: whitespace handling
check(eval_expr("  2  +  3  ") == 5.0, "whitespace")

print(f"RESULT {passed}/{total}")
