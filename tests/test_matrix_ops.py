"""Test harness for matrix_ops — imports from solution.py"""
from solution import mat_multiply, mat_transpose, mat_determinant

passed = 0
total = 12

def check(condition, name):
    global passed
    if condition:
        passed += 1
        print(f"PASS test_{name}")
    else:
        print(f"FAIL test_{name}")

check(mat_transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]], "transpose_2x2")
check(mat_transpose([[1]]) == [[1]], "transpose_1x1")
check(mat_transpose([[1, 2, 3]]) == [[1], [2], [3]], "transpose_1x3")

r = mat_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
check(r == [[19, 22], [43, 50]], "multiply_2x2")

r2 = mat_multiply([[1, 2, 3]], [[4], [5], [6]])
check(r2 == [[32]], "multiply_1x3_3x1")

try:
    mat_multiply([[1, 2]], [[1, 2]])
    check(False, "multiply_bad_dims")
except ValueError:
    check(True, "multiply_bad_dims")

check(mat_determinant([[5]]) == 5, "det_1x1")
check(mat_determinant([[1, 2], [3, 4]]) == -2, "det_2x2")

d = mat_determinant([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
check(abs(d - 27) < 0.001, "det_3x3")

try:
    mat_determinant([[1, 2], [3, 4], [5, 6]])
    check(False, "det_not_square")
except ValueError:
    check(True, "det_not_square")

check(mat_multiply([[1, 0], [0, 1]], [[5, 6], [7, 8]]) == [[5, 6], [7, 8]], "multiply_identity")
check(mat_determinant([[1, 0], [0, 1]]) == 1, "det_identity")

print(f"RESULT {passed}/{total}")
