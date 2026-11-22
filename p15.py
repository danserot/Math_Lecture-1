import numpy as np
from numpy.testing import assert_allclose


def check_unfoldings():
    T = np.arange(24).reshape(2, 3, 4)
    flat = T.reshape(-1)
    for i, j, k in np.ndindex(T.shape):
        assert T[i, j, k] == flat[12 * i + 4 * j + k]
    print("Flat indexing: all 24 entries match")

    first = T.reshape(2, 12)
    second = T.transpose(1, 0, 2).reshape(3, 8)
    first_rank = np.linalg.matrix_rank(first)
    second_rank = np.linalg.matrix_rank(second)
    assert first_rank == second_rank == 2
    print(f"First unfolding, rank {first_rank}:\n{first}")
    print(f"Second unfolding, rank {second_rank}:\n{second}")


def check_einsum(rng):
    u = rng.normal(size=4)
    v = rng.normal(size=4)
    A = rng.normal(size=(4, 4))
    B = rng.normal(size=(4, 4))

    # Standard operations matching the NumPy references in the problem.
    operations = [
        ("Dot product", np.einsum("i,i->", u, v), u @ v),
        ("Outer product", np.einsum("i,j->ij", u, v), np.outer(u, v)),
        ("Matrix-vector product", np.einsum("ij,j->i", A, u), A @ u),
        ("Matrix product", np.einsum("ij,jk->ik", A, B), A @ B),
        ("Transpose", np.einsum("ij->ji", A), A.T),
        ("Trace", np.einsum("ii->", A), np.trace(A)),
        ("Diagonal", np.einsum("ii->i", A), np.diag(A)),
        ("Elementwise product", np.einsum("ij,ij->ij", A, B), A * B),
    ]
    for name, actual, expected in operations:
        assert_allclose(actual, expected, atol=1e-12)
        print(f"{name}: OK")

    batch_A = rng.normal(size=(100, 4, 4))
    batch_B = rng.normal(size=(100, 4, 4))

    # The index b stays in the output, so each matrix pair is processed separately.
    traces = np.einsum("bii->b", batch_A)
    products = np.einsum("bij,bjk->bik", batch_A, batch_B)
    trace_loop = [np.trace(matrix) for matrix in batch_A]
    product_loop = [a @ b for a, b in zip(batch_A, batch_B)]
    assert_allclose(traces, trace_loop, atol=1e-12)
    assert_allclose(products, product_loop, atol=1e-12)
    print(f"Batch traces {traces.shape}, batch products {products.shape}: OK")


def check_kronecker(rng):
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(2, 5))
    X = rng.normal(size=(4, 5))

    direct = (A @ X @ B.T).reshape(-1)
    kronecker = np.kron(A, B) @ X.reshape(-1)
    assert_allclose(direct, kronecker, atol=1e-12)
    print("Kronecker identity: OK")

    side = 100 * 100
    entries = side * side
    memory_mb = entries * np.dtype(np.float64).itemsize / 1_000_000
    print(f"Large Kronecker product: {side:,} x {side:,}, {entries:,} entries")
    print(f"Memory for float64: {memory_mb:g} MB")


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


def check_rank_one(rng):
    u = rng.normal(size=3)
    v = rng.normal(size=4)
    w = rng.normal(size=5)
    T = rank_one(u, v, w)
    outer = np.outer(u, v)

    for k in range(len(w)):
        assert_allclose(T[:, :, k], w[k] * outer, atol=1e-12)
    print(f"Rank-one tensor {T.shape}: all slices match w[k] * outer(u, v)")


def main():
    rng = np.random.default_rng(42)
    check_unfoldings()
    check_einsum(rng)
    check_kronecker(rng)
    check_rank_one(rng)


if __name__ == "__main__":
    main()
