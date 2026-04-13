"""Test harness for sort_by_frequency — imports from solution.py"""
from solution import sort_by_frequency

passed = 0
total = 6

tests = [
    (([4, 5, 6, 5, 4, 3],), lambda r: r in [[4, 5, 4, 5, 6, 3], [5, 4, 5, 4, 6, 3]], "basic"),
    (([1, 1, 2, 2, 3],), lambda r: r in [[1, 1, 2, 2, 3], [2, 2, 1, 1, 3]], "tie"),
    (([],), lambda r: r == [], "empty"),
    (([7],), lambda r: r == [7], "single"),
    (([1, 2, 3],), lambda r: r == [1, 2, 3], "all_unique"),
    (([3, 3, 3, 1, 1, 2],), lambda r: r == [3, 3, 3, 1, 1, 2], "clear_winner"),
]

for args, check, name in tests:
    try:
        result = sort_by_frequency(*args)
        if check(result):
            passed += 1
            print(f"PASS test_{name}")
        else:
            print(f"FAIL test_{name} (got {result})")
    except Exception as e:
        print(f"FAIL test_{name} ({e})")

print(f"RESULT {passed}/{total}")
