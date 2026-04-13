"""Test harness for gradient_descent_regression — imports from solution.py"""
import numpy as np

np.random.seed(42)
from solution import LinearRegression, r_squared

passed = 0
total = 5

# Test 1: Simple linear y = 3x + 2 + noise
X = np.random.randn(100, 1) * 2
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5
model = LinearRegression(lr=0.05, epochs=2000)
model.fit(X, y)
preds = model.predict(X)
r2 = r_squared(y, preds)
if r2 > 0.90:
    passed += 1
    print(f"PASS test_simple_linear (R²={r2:.3f})")
else:
    print(f"FAIL test_simple_linear (R²={r2:.3f}, need > 0.90)")

# Test 2: Multi-feature y = 2*x1 - 3*x2 + 1
X_multi = np.random.randn(200, 2)
y_multi = 2 * X_multi[:, 0] - 3 * X_multi[:, 1] + 1 + np.random.randn(200) * 0.3
model2 = LinearRegression(lr=0.01, epochs=3000)
model2.fit(X_multi, y_multi)
preds2 = model2.predict(X_multi)
r2_multi = r_squared(y_multi, preds2)
if r2_multi > 0.90:
    passed += 1
    print(f"PASS test_multi_feature (R²={r2_multi:.3f})")
else:
    print(f"FAIL test_multi_feature (R²={r2_multi:.3f}, need > 0.90)")

# Test 3: Loss decreasing
if len(model.loss_history) == 2000 and model.loss_history[-1] < model.loss_history[0]:
    passed += 1
    print(f"PASS test_loss_decreasing ({model.loss_history[0]:.3f} -> {model.loss_history[-1]:.3f})")
else:
    print(f"FAIL test_loss_decreasing (len={len(model.loss_history)})")

# Test 4: L2 regularization
model_reg = LinearRegression(lr=0.05, epochs=2000, l2_reg=1.0)
model_reg.fit(X, y)
norm_unreg = np.sum(model.weights ** 2)
norm_reg = np.sum(model_reg.weights ** 2)
if norm_reg < norm_unreg:
    passed += 1
    print(f"PASS test_regularization (unreg={norm_unreg:.3f}, reg={norm_reg:.3f})")
else:
    print(f"FAIL test_regularization (unreg={norm_unreg:.3f}, reg={norm_reg:.3f})")

# Test 5: r_squared perfect
if abs(r_squared(np.array([1, 2, 3]), np.array([1, 2, 3])) - 1.0) < 1e-10:
    passed += 1
    print("PASS test_r_squared_perfect")
else:
    print("FAIL test_r_squared_perfect")

print(f"RESULT {passed}/{total}")
