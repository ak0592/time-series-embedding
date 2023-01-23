import pandas as pd

from time_series_embedding.time_series_data.parent_time_series_data import ParentTimeSeriesData
from time_series_embedding.config.time_series_config import param_to_data_key
from time_series_embedding.time_series_data.generate_gradually_dense_cluster import gradually_dense_cluster


class GraduallyDenseClusterData(ParentTimeSeriesData):

    def __init__(self, *args) -> None:
        super(GraduallyDenseClusterData, self).__init__(*args)
        self.data_key = param_to_data_key('GRADUALLY_DENSE', self.params)
        self.compute_data_and_color()

    def compute_data_and_color(self):
        self.df, color, text = gradually_dense_cluster(**self.params)
        self.n_steps = int(self.df['step'].max() + 1)
        self.n_data_points = self.df.query('step == 0').shape[0]
        self.sub_info = pd.DataFrame(data=[color, text]).T.set_axis(['color', 'text'], axis='columns')
