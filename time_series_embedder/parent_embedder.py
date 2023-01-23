from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from time_series_embedding.time_series_data.parent_time_series_data import ParentTimeSeriesData
from typing import List, Dict, Union


class ParentTimeSeriesEmbedder(metaclass=ABCMeta):

    def __init__(self, data: ParentTimeSeriesData, embedder_params: Dict[str, Union[int, str]]) -> None:
        self.data = data
        self.params = embedder_params
        self.df = data.df
        self.n_steps = self.data.n_steps
        self.n_data_points = self.data.n_data_points
        self.sub_info = data.sub_info

    def embed(self, use_cache=False, **kwargs) -> None:
        if self.data.exists_coords(self.class_key) and self.data.exists_sub_info(self.class_key) and use_cache:
            self.coordinates = self.data.load_coords(self.class_key)
            self.sub_info = self.data.load_sub_info(self.class_key)
            self.em = {'coordinate': self.coordinates, 'sub_info': self.sub_info}
        else:
            self.exec_embed(**kwargs)
            self.em = {'coordinate': self.coordinates, 'sub_info': self.sub_info}
            self.data.save_coords(self.class_key, self.coordinates)
            self.data.save_sub_info(self.class_key, self.sub_info)

    @abstractmethod
    def exec_embed(self):
        pass


def transform_dataframe_to_list_array_and_check_original_data(n_steps, n_data_points, original_df: pd.DataFrame) -> List[np.ndarray]:
    original_data_list = []

    for step in range(n_steps):
        assert n_data_points == original_df.query(f'step == {step}').shape[0], ValueError(
            f'The number of data points in {step} step does not match with others.')
        original_data_list.append(original_df.query(f'step == {step}').drop(columns='step').to_numpy(copy=True))

    return original_data_list
