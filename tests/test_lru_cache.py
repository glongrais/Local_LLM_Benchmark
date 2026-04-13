"""Test harness for lru_cache — imports LRUCache from solution.py"""
from solution import LRUCache

passed = 0
total = 13

def check(condition, name, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name} ({detail})")

c = LRUCache(3)
c.put("a", 1)
c.put("b", 2)
c.put("c", 3)
check(c.get("a") == 1, "get_a")
check(c.size() == 3, "size_3", f"got {c.size()}")
check(c.keys()[0] == "a", "mru_after_get", f"got {c.keys()}")

c.put("d", 4)  # should evict "b"
check(c.get("b") is None, "evicted_b")
check(c.size() == 3, "size_after_evict", f"got {c.size()}")
check(c.get("c") == 3, "get_c")
check(c.get("d") == 4, "get_d")

c.put("a", 10)  # update existing
check(c.get("a") == 10, "updated_a")
check(c.size() == 3, "size_after_update", f"got {c.size()}")

# Keys order: most recent first
keys = c.keys()
check(keys[0] == "a", "mru_is_a", f"got {keys}")

c2 = LRUCache(1)
c2.put("x", 1)
c2.put("y", 2)
check(c2.get("x") is None, "cap1_evict")
check(c2.get("y") == 2, "cap1_get_y")
check(c2.size() == 1, "cap1_size", f"got {c2.size()}")

print(f"RESULT {passed}/{total}")
