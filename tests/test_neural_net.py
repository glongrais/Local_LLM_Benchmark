"""Test harness for neural_net_from_scratch — imports MLP from solution.py"""
import sys
import numpy as np

np.random.seed(42)
from solution import MLP

passed = 0
total = 3

# Test 1: XOR problem
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)
model = MLP([2, 16, 16, 1])
model.train(X_xor, y_xor, epochs=2000, lr=0.1)
preds = model.predict(X_xor)
xor_acc = np.mean(preds == y_xor)
if xor_acc >= 0.75:
    passed += 1
    print(f"PASS test_xor (acc={xor_acc:.2f})")
else:
    print(f"FAIL test_xor (acc={xor_acc:.2f}, need >= 0.75)")

# Test 2: Circular decision boundary
np.random.seed(42)
n = 200
angles = np.random.uniform(0, 2 * np.pi, n)
radii_inner = np.random.uniform(0, 1, n // 2)
radii_outer = np.random.uniform(1.5, 2.5, n // 2)
X_inner = np.column_stack([radii_inner * np.cos(angles[: n // 2]), radii_inner * np.sin(angles[: n // 2])])
X_outer = np.column_stack([radii_outer * np.cos(angles[n // 2 :]), radii_outer * np.sin(angles[n // 2 :])])
X_circle = np.vstack([X_inner, X_outer])
y_circle = np.vstack([np.zeros((n // 2, 1)), np.ones((n // 2, 1))])
model2 = MLP([2, 32, 32, 1])
model2.train(X_circle, y_circle, epochs=3000, lr=0.01)
preds2 = model2.predict(X_circle)
circle_acc = np.mean(preds2 == y_circle)
if circle_acc >= 0.80:
    passed += 1
    print(f"PASS test_circle (acc={circle_acc:.2f})")
else:
    print(f"FAIL test_circle (acc={circle_acc:.2f}, need >= 0.80)")

# Test 3: Output shapes
try:
    assert model.forward(X_xor).shape == (4, 1)
    assert model.predict(X_xor).shape == (4, 1)
    passed += 1
    print("PASS test_shapes")
except AssertionError as e:
    print(f"FAIL test_shapes ({e})")

print(f"RESULT {passed}/{total}")
