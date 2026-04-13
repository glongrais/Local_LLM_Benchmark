"""Test harness for graph_algorithms — imports from solution.py"""
from solution import has_cycle, shortest_path, topological_sort

passed = 0
total = 11

def check(condition, name, detail=""):
    global passed
    if condition:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name} ({detail})")

g1 = {"a": ["b"], "b": ["c"], "c": []}
g2 = {"a": ["b"], "b": ["c"], "c": ["a"]}
g3 = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}

check(has_cycle(g1) == False, "no_cycle")
check(has_cycle(g2) == True, "has_cycle")
check(has_cycle(g3) == False, "diamond_no_cycle")

check(shortest_path(g1, "a", "c") == ["a", "b", "c"], "path_abc")
check(shortest_path(g1, "c", "a") is None, "no_path")

sp = shortest_path(g3, "a", "d")
check(sp in [["a", "b", "d"], ["a", "c", "d"]], "diamond_path", f"got {sp}")

check(shortest_path({"a": []}, "a", "a") == ["a"], "self_path")

ts = topological_sort(g3)
check(ts.index("a") < ts.index("b") and ts.index("a") < ts.index("c"), "topo_order_a")
check(ts.index("b") < ts.index("d"), "topo_order_bd")

try:
    topological_sort(g2)
    check(False, "topo_cycle_error")
except ValueError:
    check(True, "topo_cycle_error")

check(len(topological_sort(g1)) == 3, "topo_length")

print(f"RESULT {passed}/{total}")
