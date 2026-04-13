"""Test harness for decision_tree — imports DecisionTree from solution.py"""
import numpy as np

np.random.seed(42)
from solution import DecisionTree

passed = 0
total = 5

# Test 1: Linearly separable
X_simple = np.array([[1, 1], [2, 2], [3, 3], [7, 7], [8, 8], [9, 9]], dtype=float)
y_simple = np.array([0, 0, 0, 1, 1, 1])
tree = DecisionTree(max_depth=5)
tree.fit(X_simple, y_simple)
preds = tree.predict(X_simple)
if np.all(preds == y_simple):
    passed += 1
    print("PASS test_linear")
else:
    print(f"FAIL test_linear ({preds} vs {y_simple})")

# Test 2: XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
tree2 = DecisionTree(max_depth=5)
tree2.fit(X_xor, y_xor)
preds2 = tree2.predict(X_xor)
xor_acc = np.mean(preds2 == y_xor)
if xor_acc == 1.0:
    passed += 1
    print(f"PASS test_xor (acc={xor_acc:.2f})")
else:
    print(f"FAIL test_xor (acc={xor_acc:.2f})")

# Test 3: Multi-class
np.random.seed(42)
X_multi = np.vstack([
    np.random.randn(40, 3) + np.array([0, 0, 0]),
    np.random.randn(40, 3) + np.array([5, 5, 5]),
    np.random.randn(40, 3) + np.array([10, 0, 5]),
])
y_multi = np.array([0] * 40 + [1] * 40 + [2] * 40)
tree3 = DecisionTree(max_depth=10)
tree3.fit(X_multi, y_multi)
preds3 = tree3.predict(X_multi)
multi_acc = np.mean(preds3 == y_multi)
if multi_acc >= 0.90:
    passed += 1
    print(f"PASS test_multiclass (acc={multi_acc:.2f})")
else:
    print(f"FAIL test_multiclass (acc={multi_acc:.2f}, need >= 0.90)")

# Test 4: Depth constraint
tree_stump = DecisionTree(max_depth=1)
tree_stump.fit(X_multi, y_multi)
preds_stump = tree_stump.predict(X_multi)
stump_acc = np.mean(preds_stump == y_multi)
if stump_acc < multi_acc or stump_acc >= 0.5:
    passed += 1
    print(f"PASS test_stump (acc={stump_acc:.2f} < full={multi_acc:.2f})")
else:
    print(f"FAIL test_stump (acc={stump_acc:.2f})")

# Test 5: Predict shape
if tree3.predict(X_multi[:5]).shape == (5,):
    passed += 1
    print("PASS test_shape")
else:
    print(f"FAIL test_shape (got {tree3.predict(X_multi[:5]).shape})")

print(f"RESULT {passed}/{total}")
