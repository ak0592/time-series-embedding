import pandas as pd
import os
import hashlib

from abc import ABCMeta, abstractmethod
from typing import Dict, Union


class ParentTimeSeriesData(metaclass=ABCMeta):

    def __init__(self, data_path: str, cache_path: str, data_params: Dict[str, Union[int, str]]) -> None:
        self.data_path = data_path
        self.cache_path = cache_path
        self.params = data_params
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

    def load_coords(self, class_key: str) -> pd.DataFrame:
        file_name = f'{class_key}_{self.data_key}_coords'
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()
        return pd.read_csv(os.path.join(self.cache_path, f"{file_name_hash}.csv"))

    def load_sub_info(self, class_key: str) -> pd.DataFrame:
        file_name = f'{class_key}_{self.data_key}_sub_info'
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()
        return pd.read_csv(os.path.join(self.cache_path, f"{file_name_hash}.csv"))

    def exists_coords(self, class_key: str) -> bool:
        file_name = f'{class_key}_{self.data_key}_coords'
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()
        return os.path.exists(os.path.join(self.cache_path, f"{file_name_hash}.csv"))

    def exists_sub_info(self, class_key: str) -> bool:
        file_name = f'{class_key}_{self.data_key}_sub_info'
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()
        return os.path.exists(os.path.join(self.cache_path, f"{file_name_hash}.csv"))

    def save_coords(self, class_key: str, coord_df: pd.DataFrame) -> None:
        file_name = f'{class_key}_{self.data_key}_coords'
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()
        coord_df.to_csv(os.path.join(self.cache_path, f"{file_name_hash}.csv"), index=False)

    def save_sub_info(self, class_key, sub_info_df: pd.DataFrame) -> None:
        file_name = f'{class_key}_{self.data_key}_sub_info'
        file_name_hash = hashlib.sha256(file_name.encode()).hexdigest()
        sub_info_df.to_csv(os.path.join(self.cache_path, f"{file_name_hash}.csv"), index=False)

    @abstractmethod
    def compute_data_and_color(self):
        pass
