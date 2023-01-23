from time_series_embedding.time_series_data.parent_time_series_data import ParentTimeSeriesData
from time_series_embedding.time_series_data.const import TIME_SERIES_CACHE_PATH, TIME_SERIES_DATA_PATH, TIME_SERIES_DATA_REF
from time_series_embedding.config.time_series_config import TimeSeriesConfig
from time_series_embedding.config.const import DEFAULT_DATA_YAML_FILEPATH, TEST_DATA_YAML_FILEPATH


def choose_time_series_data(key: str, is_test=False) -> ParentTimeSeriesData:
    if key not in TIME_SERIES_DATA_REF:
        raise NotImplementedError(f"{key} is not supported")
    data_config_file_path = DEFAULT_DATA_YAML_FILEPATH if not is_test else TEST_DATA_YAML_FILEPATH
    data_params = TimeSeriesConfig.load_params(data_config_file_path)[key]

    return TIME_SERIES_DATA_REF[key](TIME_SERIES_DATA_PATH, TIME_SERIES_CACHE_PATH, data_params)
