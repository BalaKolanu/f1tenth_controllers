import numpy as np


def nearest_point(point: np.ndarray, trajectory: np.ndarray):
    """Return nearest projection on piecewise linear trajectory."""
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    l2s = np.maximum(l2s, 1e-12)

    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])

    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0

    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        delta = point - projections[i]
        dists[i] = np.sqrt(np.sum(delta * delta))

    min_dist_segment = int(np.argmin(dists))
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment
