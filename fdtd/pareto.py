"""Pareto front (minimise both error and cost)."""


def front(points):
    """points: list of (cost, error). Returns indices of non-dominated points, sorted by cost."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    kept, best_err = [], float("inf")
    for i in order:
        if points[i][1] < best_err:
            kept.append(i)
            best_err = points[i][1]
    return kept
