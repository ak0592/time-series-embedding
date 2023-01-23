import pandas as pd
import numpy as np

from time_series_embedding.time_series_data.parent_time_series_data import ParentTimeSeriesData
from time_series_embedding.config.time_series_config import param_to_data_key


class TimeSeriesSwissRollData(ParentTimeSeriesData):

    def __init__(self, *args) -> None:
        super(TimeSeriesSwissRollData, self).__init__(*args)
        self.data_key = param_to_data_key('TIME_SERIES_SWISSROLL', self.params)
        self.compute_data_and_color()

    def compute_data_and_color(self):
        self.df, color, text = make_time_series_swissroll(**self.params)
        self.n_steps = int(self.df['step'].max() + 1)
        self.n_data_points = self.df.query('step == 0').shape[0]
        self.sub_info = pd.DataFrame(data=[color, text]).T.set_axis(['color', 'text'], axis='columns')


def make_time_series_swissroll(n_data_points=1000, n_steps=5):
    base_data_points = np.random.random_sample((n_data_points, 2))
    base_data_points[:, 0] = base_data_points[:, 0] * 10
    base_data_points[:, 1] = base_data_points[:, 1] * 3

    color_ind = np.argmax(base_data_points.max(axis=0) - base_data_points.min(axis=0))
    base_data_points = base_data_points[np.argsort(base_data_points[:, color_ind], axis=0)]

    entrainment_ratios = np.linspace(1.5, 0.5, n_steps)
    all_step_data_points = []
    for step in range(n_steps):
        all_step_data_points.append(add_new_swissroll_dimension(base_data_points, 0, entrainment_ratios[step]))
    all_step_df = pd.DataFrame(np.concatenate(all_step_data_points, axis=0), columns=[f'col{i}' for i in range(all_step_data_points[0].shape[-1])])
    all_step_df['step'] = np.repeat(np.arange(n_steps), n_data_points)
    colors = np.arange(n_data_points)
    text = [f'{i}' for i in range(n_data_points)]

    return all_step_df, colors, text


def add_new_swissroll_dimension(points, parameter_dim=0, entrainment_rate=1.5):
    if parameter_dim is None:
        parameter_dim = np.random.choice(np.arange(points.shape[1]))
    standardized_points = standardize(points, var=1.0)
    t = entrainment_rate * (np.pi + 2 * np.arccos(normalize(standardized_points)[:, parameter_dim]))

    standardized_points[:, parameter_dim] = t * np.cos(t)
    new_points = t * np.sin(t)
    return np.concatenate((standardized_points, np.expand_dims(new_points, 1)), axis=1)


def normalize(points):
    normal_points = points - points.mean(axis=0)

    return normal_points / np.abs(normal_points).max(axis=0)


def standardize(points, var=1.0):
    std = np.std(points, axis=0)
    mean = np.mean(points, axis=0)

    return (points - mean) * np.sqrt(var) / std
