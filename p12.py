from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def dual_basis(P):
    """Return the dual basis as rows."""
    return np.linalg.inv(P)


def lagrange_coefficients(nodes):
    """Return the coefficients of each Lagrange polynomial in a column."""
    V = np.vander(nodes, increasing=True)
    return np.linalg.inv(V)


def interpolate(nodes, values, t):
    """Evaluate p(t) = sum(values[k] * L_k(t))."""
    t = np.asarray(t, dtype=float)
    p = np.zeros_like(t)

    for k, node in enumerate(nodes):
        # L_k is 1 at nodes[k] and 0 at the other nodes.
        L = np.ones_like(t)
        for j, other in enumerate(nodes):
            if j != k:
                L *= (t - other) / (node - other)
        p += values[k] * L

    return p


def plot_interpolation(ax, f, interval, count, title):
    left, right = interval
    nodes = np.linspace(left, right, count)
    values = f(nodes)
    np.testing.assert_allclose(interpolate(nodes, values, nodes), values)

    t = np.linspace(left, right, 2001)
    original = f(t)
    approximation = interpolate(nodes, values, t)
    ax.plot(t, original, color="#1d4ed8", linewidth=2.4, label="f(t)")
    ax.plot(t, approximation, "--", color="#b45309", linewidth=1.7, label="p(t)")
    ax.scatter(nodes, values, s=22, facecolor="white", edgecolor="#1d4ed8",
               zorder=3, label="Узлы")
    ax.set(title=title, xlabel="t")
    ax.margins(x=0.025)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    return np.max(np.abs(original - approximation))


def main():
    rng = np.random.default_rng(42)
    P = rng.normal(size=(5, 5))
    x = rng.normal(size=5)
    R = dual_basis(P)
    coordinates = R @ x
    assert np.allclose(R @ P, np.eye(5))
    assert np.allclose(P @ coordinates, x)
    print("Dual basis: R @ P = I and P @ (R @ x) = x")

    coefficients = lagrange_coefficients([-1, 0, 1])
    expected = [[0, 1, 0], [-0.5, 0, 0.5], [0.5, -1, 0.5]]
    np.testing.assert_allclose(coefficients, expected)
    print("Lagrange coefficients in columns (powers 1, t, t^2):")
    print(coefficients)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    fig.suptitle("Интерполяция Лагранжа", fontsize=16)
    sine_error = plot_interpolation(
        ax1, lambda t: np.sin(3 * t), (-1, 1), 9, r"$\sin(3t)$, 9 узлов"
    )
    runge_error = plot_interpolation(
        ax2, lambda t: 1 / (1 + t**2), (-5, 5), 21, r"$1/(1+t^2)$, 21 узел"
    )
    print(f"Max error on the plotted grid, sin(3t): {sine_error:.4g}")
    print(f"Max error on the plotted grid, Runge function: {runge_error:.4g}")

    fig.savefig(Path(__file__).with_suffix(".png"), dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
