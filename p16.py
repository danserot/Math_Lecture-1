from timeit import repeat

import numpy as np
from numpy.testing import assert_allclose


def lu(A):
    """Return L, U with A = L @ U, using no row swaps."""
    U = np.array(A, dtype=float, copy=True)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("A must be a square matrix")

    n = U.shape[0]
    L = np.eye(n)

    for k in range(n):
        pivot = U[k, k]
        if pivot == 0:
            raise np.linalg.LinAlgError(f"Zero pivot in column {k}")

        multipliers = U[k + 1:, k] / pivot
        L[k + 1:, k] = multipliers
        U[k + 1:, k + 1:] -= np.outer(multipliers, U[k, k + 1:])
        U[k + 1:, k] = 0

    return L, U


def solve_lu(L, U, b):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    # Forward substitution: L @ y = b.
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Back substitution: U @ x = y.
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    return x


def check_lu(rng):
    # A substitute for Example 6.107, whose matrix is absent from the screenshot.
    A = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]
    L, U = lu(A)
    assert_allclose(L, [[1, 0, 0], [2, 1, 0], [-1, -1, 1]])
    assert_allclose(U, [[2, 1, 1], [0, -8, -2], [0, 0, 1]])
    assert_allclose(L @ U, A)

    for _ in range(20):
        A = rng.normal(size=(6, 6))
        b = rng.normal(size=6)
        L, U = lu(A)
        assert_allclose(L @ U, A, atol=1e-10)
        assert_allclose(solve_lu(L, U, b), np.linalg.solve(A, b), atol=1e-10)

    print("LU factorization and solve: 20 random 6 x 6 tests passed")


def show_small_pivot():
    A = np.array([[1e-17, 1], [1, 1]])
    b = np.array([1, 2])

    L, U = lu(A)
    print("Small pivot, no row swap:", solve_lu(L, U, b))

    L, U = lu(A[::-1])
    with_swap = solve_lu(L, U, b[::-1])
    expected = np.linalg.solve(A, b)
    assert_allclose(with_swap, expected)
    print("With row swap:", with_swap)
    print("NumPy solve:", expected)


def gram_inverse(X):
    return np.linalg.inv(X.T @ X)


def update_inverse(inverse, row):
    # Adding a row changes the Gram matrix by outer(row, row).
    z = inverse @ row
    return inverse - np.outer(z, z) / (1 + row @ z)


def compare_updates(rng):
    X = rng.normal(size=(200, 50))
    new_rows = rng.normal(size=(100, 50))
    all_rows = np.vstack((X, new_rows))
    initial = gram_inverse(X)

    def update_each():
        inverse = initial
        for row in new_rows:
            inverse = update_inverse(inverse, row)
        return inverse

    def recompute_each():
        for size in range(len(X) + 1, len(all_rows) + 1):
            inverse = gram_inverse(all_rows[:size])
        return inverse

    updated = update_each()
    expected = gram_inverse(all_rows)
    assert_allclose(updated, expected, rtol=1e-10, atol=1e-12)
    print(f"Max difference from final inverse: {np.max(np.abs(updated - expected)):.3g}")

    methods = [
        ("100 Sherman-Morrison updates", update_each),
        ("100 full recomputations", recompute_each),
        ("One final recomputation", lambda: gram_inverse(all_rows)),
    ]
    for name, method in methods:
        seconds = min(repeat(method, number=1, repeat=5))
        print(f"{name}: {seconds * 1000:.3f} ms (best of 5)")


def main():
    rng = np.random.default_rng(42)
    check_lu(rng)
    show_small_pivot()
    compare_updates(rng)


if __name__ == "__main__":
    main()
