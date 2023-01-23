import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from typing import List


def gradually_dense_cluster(n_cluster: int = 10, n_dim: int = 100, n_points_per_cluster: int = 200, mean: float = 1.0,
                            variance: float = 0.1, n_steps: int = 4, advection_ratio: float = 0.5,
                            random_state: int = None) -> (pd.DataFrame, np.ndarray, List[str]):
    if n_dim < n_cluster:
        raise ValueError('n_dim must be larger than n_cluster.')

    random_state = check_random_state(random_state)
    # decide initial points state
    all_points_per_step = []

    indices = random_state.permutation(n_dim)[0:n_cluster]
    all_cluster_means = []
    # create initial point coordinate
    for c in range(n_cluster):
        means = np.zeros(n_dim)
        means[indices[c]] = mean  # indicesで指定されるindexのmeanだけ１、残りは０
        all_cluster_means.append(means)

        # N(mean, 0.1)のn_dim次元正規分布 size:(n_points_per_cluster, n_dim)
        all_points_per_step.append(
            random_state.multivariate_normal(means, np.eye(n_dim) * variance, n_points_per_cluster))
    all_points_per_step = np.concatenate(all_points_per_step)

    # create sub info
    color = np.concatenate([[i] * n_points_per_cluster for i in range(n_cluster)])  # class labels
    text = []
    for i_cluster in range(n_cluster):
        for j_points in range(n_points_per_cluster):
            text.append(f'{i_cluster}_{j_points}')

    # decide points state changing
    all_step_data_points = [np.array(all_points_per_step)]
    for step in range(n_steps - 1):
        points_next_dtep = np.array(all_step_data_points[step])
        for c in range(n_cluster):
            start, end = n_points_per_cluster * c, n_points_per_cluster * (c + 1)
            points_next_dtep[start: end] += advection_ratio * (
                        all_cluster_means[c] - points_next_dtep[start: end])  # stepごとに平均に近づいていく

        all_step_data_points.append(points_next_dtep)

    # add step axis
    all_step_df = []
    for s in range(n_steps):
        step = np.array([s] * n_cluster * n_points_per_cluster)
        all_step_df.append(
            np.concatenate((all_step_data_points[s], np.expand_dims(step, axis=1)), axis=1))
    all_step_df = np.array(all_step_df).reshape(-1, n_dim + 1)

    columns = [f'col{i}' for i in range(n_dim)]
    columns.append('step')

    all_step_df = pd.DataFrame(data=all_step_df, columns=columns)

    return all_step_df, color, text
