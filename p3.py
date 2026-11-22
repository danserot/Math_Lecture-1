import numpy as np


def rref(A, tol=1e-10):
    R = np.array(A, dtype=float, copy=True)
    rows, cols = R.shape
    pivots = []
    row = 0

    for col in range(cols):
        if row == rows:
            break

        # Find the largest absolute entry among the remaining rows.
        best = row + np.argmax(np.abs(R[row:, col]))
        if abs(R[best, col]) < tol:
            R[row:, col] = 0
            continue

        R[[row, best]] = R[[best, row]]
        R[row] /= R[row, col]

        for i in range(rows):
            if i != row:
                R[i] -= R[i, col] * R[row]

        pivots.append(col)
        row += 1

    return R, pivots


def rank(A):
    return len(rref(A)[1])


def null_space_basis(A):
    R, pivots = rref(A)
    cols = R.shape[1]
    basis = []

    for col in range(cols):
        if col in pivots:
            continue

        # Set one free variable to 1 and the other free variables to 0.
        v = np.zeros(cols)
        v[col] = 1
        for row, pivot in enumerate(pivots):
            v[pivot] = -R[row, col]
        basis.append(v)

    return basis


def col_space_basis(A):
    A = np.asarray(A)
    _, pivots = rref(A)
    return [A[:, col].copy() for col in pivots]


def main():
    A = np.array([[1, 2, -1], [3, -1, 4]])
    R, pivots = rref(A)
    assert np.allclose(R, [[1, 0, 1], [0, 1, -1]])
    assert pivots == [0, 1]
    assert np.allclose(null_space_basis(A), [[-1, 1, 1]])
    assert np.array_equal(col_space_basis(A), [[1, 3], [2, -1]])
    print("RREF:")
    print(R)
    print("Pivot columns:", pivots)

    # Substitute with the expected RREF of F from Example 6.31.
    F_test = [[1, 2, 3], [3, -1, -5], [4, 1, -2]]
    assert np.allclose(rref(F_test)[0], [[1, 0, -1], [0, 1, 2], [0, 0, 0]])

    rng = np.random.default_rng(42)
    for _ in range(20):
        A = rng.integers(-9, 10, size=(5, 7))
        basis = null_space_basis(A)
        assert all(np.allclose(A @ v, 0) for v in basis)
        assert rank(A) + len(basis) == 7
        assert rank(A) == np.linalg.matrix_rank(A)
    print("20 random 5 x 7 tests passed")

    B = rng.normal(size=(6, 4))
    C = rng.normal(size=(4, 8))
    A = B @ C
    basis = null_space_basis(A)
    assert rank(A) == 4 and len(basis) == 4
    assert all(np.allclose(A @ v, 0) for v in basis)
    print("6 x 8 product: rank = 4, nullity = 4")


if __name__ == "__main__":
    main()
