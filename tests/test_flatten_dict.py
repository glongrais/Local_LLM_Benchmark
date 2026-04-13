"""Test harness for flatten_dict — imports from solution.py"""
from solution import flatten_dict

passed = 0
total = 6

tests = [
    ({"a": {"b": 1, "c": {"d": 2}}, "e": 3}, {}, {"a.b": 1, "a.c.d": 2, "e": 3}, "nested"),
    ({}, {}, {}, "empty"),
    ({"x": 1}, {}, {"x": 1}, "flat"),
    ({"a": {"b": {"c": {"d": 1}}}}, {}, {"a.b.c.d": 1}, "deep"),
    ({"a": {"b": 1}}, {"sep": "/"}, {"a/b": 1}, "custom_sep"),
    ({"a": {"b": [1, 2]}}, {}, {"a.b": [1, 2]}, "list_value"),
]

for inp, kwargs, expected, name in tests:
    try:
        result = flatten_dict(inp, **kwargs)
        if result == expected:
            passed += 1
            print(f"PASS test_{name}")
        else:
            print(f"FAIL test_{name} (got {result})")
    except Exception as e:
        print(f"FAIL test_{name} ({e})")

print(f"RESULT {passed}/{total}")
