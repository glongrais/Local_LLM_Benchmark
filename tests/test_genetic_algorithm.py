"""Test harness for genetic_algorithm — imports GeneticAlgorithm from solution.py"""
import numpy as np

np.random.seed(42)
from solution import GeneticAlgorithm

passed = 0
total = 3


# Test 1: Sphere (maximize negative sum of squares -> optimum at 0)
def neg_sphere(x):
    return -np.sum(x**2)


ga = GeneticAlgorithm(pop_size=100, n_genes=5, mutation_rate=0.1)
best, fitness = ga.optimize(neg_sphere, generations=300, gene_range=(-5.0, 5.0))
if fitness > -1.0 and best.shape == (5,):
    passed += 1
    print(f"PASS test_sphere (fitness={fitness:.4f})")
else:
    print(f"FAIL test_sphere (fitness={fitness:.4f}, shape={best.shape})")


# Test 2: Rastrigin-like (harder, local optima)
def neg_rastrigin(x):
    A = 10
    return -(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


ga2 = GeneticAlgorithm(pop_size=200, n_genes=3, mutation_rate=0.15)
best2, fitness2 = ga2.optimize(neg_rastrigin, generations=500, gene_range=(-5.0, 5.0))
if fitness2 > -15.0:
    passed += 1
    print(f"PASS test_rastrigin (fitness={fitness2:.4f})")
else:
    print(f"FAIL test_rastrigin (fitness={fitness2:.4f}, need > -15.0)")


# Test 3: Single variable (peak at x=3)
def single_peak(x):
    return -((x[0] - 3.0) ** 2)


ga3 = GeneticAlgorithm(pop_size=50, n_genes=1, mutation_rate=0.1)
best3, fitness3 = ga3.optimize(single_peak, generations=200, gene_range=(-10.0, 10.0))
if abs(best3[0] - 3.0) < 1.0 and fitness3 > -1.0:
    passed += 1
    print(f"PASS test_single_peak (x={best3[0]:.2f}, fitness={fitness3:.4f})")
else:
    print(f"FAIL test_single_peak (x={best3[0]:.2f}, fitness={fitness3:.4f})")

print(f"RESULT {passed}/{total}")
