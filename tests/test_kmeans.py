"""Test harness for kmeans_from_scratch — imports KMeans from solution.py"""
import numpy as np
from collections import Counter

np.random.seed(42)
from solution import KMeans

passed = 0
total = 4

# Test 1: Well-separated clusters
c1 = np.random.randn(50, 2) + np.array([0, 0])
c2 = np.random.randn(50, 2) + np.array([10, 10])
c3 = np.random.randn(50, 2) + np.array([10, 0])
X = np.vstack([c1, c2, c3])
true_labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

km = KMeans(k=3)
km.fit(X)
preds = km.predict(X)

cluster_map = {}
for true_label in range(3):
    mask = true_labels == true_label
    pred_labels = preds[mask]
    most_common = Counter(pred_labels.tolist()).most_common(1)[0][0]
    cluster_map[true_label] = most_common
mapped = np.array([cluster_map[t] for t in true_labels])
acc = np.mean(mapped == preds)
if acc >= 0.90:
    passed += 1
    print(f"PASS test_clusters (acc={acc:.2f})")
else:
    print(f"FAIL test_clusters (acc={acc:.2f}, need >= 0.90)")

# Test 2: Attributes
try:
    assert km.centroids.shape == (3, 2), f"Wrong centroid shape: {km.centroids.shape}"
    assert isinstance(km.inertia, (float, np.floating)), f"Inertia should be float"
    assert km.inertia > 0
    passed += 1
    print("PASS test_attributes")
except (AssertionError, AttributeError) as e:
    print(f"FAIL test_attributes ({e})")

# Test 3: 1D data
X_1d = np.array([[1], [2], [1.5], [10], [11], [10.5]], dtype=float)
km2 = KMeans(k=2)
km2.fit(X_1d)
p2 = km2.predict(X_1d)
if p2[0] == p2[1] == p2[2] and p2[3] == p2[4] == p2[5] and p2[0] != p2[3]:
    passed += 1
    print("PASS test_1d")
else:
    print(f"FAIL test_1d (labels={p2.tolist()})")

# Test 4: Predict on new data
new_point = np.array([[5, 5]])
pred_new = km.predict(new_point)
if pred_new.shape == (1,):
    passed += 1
    print("PASS test_predict_new")
else:
    print(f"FAIL test_predict_new (shape={pred_new.shape})")

print(f"RESULT {passed}/{total}")
