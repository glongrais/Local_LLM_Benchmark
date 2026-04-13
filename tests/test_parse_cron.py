"""Test harness for parse_cron — imports from solution.py"""
from solution import parse_cron

passed = 0
total = 10

# Test group 1: */15 0 1,15 * 1-5
r = parse_cron("*/15 0 1,15 * 1-5")
checks = [
    (r["minute"] == [0, 15, 30, 45], "minute_step"),
    (r["hour"] == [0], "hour_single"),
    (r["day_of_month"] == [1, 15], "dom_list"),
    (r["month"] == list(range(1, 13)), "month_wildcard"),
    (r["day_of_week"] == [1, 2, 3, 4, 5], "dow_range"),
]
for check, name in checks:
    if check:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name}")

# Test group 2: 0 0 * * *
r2 = parse_cron("0 0 * * *")
checks2 = [
    (r2["minute"] == [0], "midnight_minute"),
    (r2["hour"] == [0], "midnight_hour"),
    (r2["day_of_month"] == list(range(1, 32)), "midnight_dom"),
]
for check, name in checks2:
    if check:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name}")

# Test group 3: 5,10,15 */6 * * *
r3 = parse_cron("5,10,15 */6 * * *")
checks3 = [
    (r3["minute"] == [5, 10, 15], "list_minute"),
    (r3["hour"] == [0, 6, 12, 18], "step_hour"),
]
for check, name in checks3:
    if check:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name}")

print(f"RESULT {passed}/{total}")
