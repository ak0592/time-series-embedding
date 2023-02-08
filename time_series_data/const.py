from os import path
from os.path import join
from time_series_data.gradually_dense_cluster_data import GraduallyDenseClusterData
from time_series_data.time_series_swissroll import TimeSeriesSwissRollData


# dir_path = 'Your path to time-series-embedding'
dir_path = '/Users/homeakira/time-series-embedding'

TIME_SERIES_DATA_PATH = join(path.sep, dir_path, 'time_series_data', 'files')
TIME_SERIES_CACHE_PATH = join(path.sep, dir_path, 'cache')
TIME_SERIES_DATA_REF = {
        'GRADUALLY_DENSE': GraduallyDenseClusterData,
        'TIME_SERIES_SWISSROLL': TimeSeriesSwissRollData
        }
