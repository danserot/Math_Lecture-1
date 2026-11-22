from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


SQUARE = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
LETTER_F = np.array([
    (0.2, 0.1), (0.4, 0.1), (0.4, 0.45), (0.65, 0.45), (0.65, 0.6),
    (0.4, 0.6), (0.4, 0.75), (0.8, 0.75), (0.8, 0.9), (0.2, 0.9),
])


def signed_area(points):
    x, y = np.asarray(points).T
    # Shift by one vertex, wrapping from the last vertex to the first.
    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)
    return 0.5 * np.sum(x * next_y - y * next_x)


def draw_panel(ax, A, title):
    A = np.asarray(A)
    ax.axhline(0, color="#b9c2ce", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#b9c2ce", linewidth=0.8, zorder=0)

    for t in (0.25, 0.5, 0.75):
        vertical = np.array([(t, 0), (t, 1)]) @ A.T
        horizontal = np.array([(0, t), (1, t)]) @ A.T
        ax.plot(*vertical.T, color="#cbd5e1", linewidth=0.9)
        ax.plot(*horizontal.T, color="#cbd5e1", linewidth=0.9)

    ax.add_patch(Polygon(SQUARE @ A.T, fill=False, edgecolor="#475569",
                         linewidth=1.7, zorder=3))
    ax.add_patch(Polygon(LETTER_F @ A.T, facecolor="#93c5fd",
                         edgecolor="#1d4ed8", linewidth=1.7, zorder=4))
    ax.autoscale_view()
    ax.margins(0.15)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=12)


def plot_transformation(A):
    """Draw the unit square, grid, and letter F before and after A."""
    fig, (before, after) = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    draw_panel(before, np.eye(2), "До преобразования")
    draw_panel(after, A, "После преобразования")
    return fig


def main():
    # Figure 2 is absent from the screenshot, so use four example matrices.
    examples = [
        ("Поворот на 90°", [[0, -1], [1, 0]]),
        ("Растяжение вдоль x", [[2, 0], [0, 1]]),
        ("Скос вдоль x", [[1, 1], [0, 1]]),
        ("Отражение относительно y", [[-1, 0], [0, 1]]),
        ("Вырожденная матрица", [[1, 2], [2, 4]]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), layout="constrained")
    fig.suptitle("Линейные преобразования единичного квадрата", fontsize=16)
    draw_panel(axes.flat[0], np.eye(2), "Исходный квадрат")
    for ax, (title, A) in zip(axes.flat[1:], examples):
        determinant = np.linalg.det(A)
        draw_panel(ax, A, f"{title}\n$\\det A = {determinant:g}$")

    area_examples = examples + [("Матрица с рис. 4", [[3, 1], [1, 2]])]
    for title, A in area_examples:
        A = np.array(A)
        area = signed_area(SQUARE @ A.T)
        determinant = np.linalg.det(A)
        if not np.isclose(area, determinant):
            raise ValueError(f"Площадь и определитель не совпали для {title}")
        print(f"{title}: площадь со знаком = {area:g}, det(A) = {determinant:g}")

    output = Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
