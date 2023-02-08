import os

# dir_path = 'Your path to time-series-embedding'
dir_path = '/Users/homeakira/time-series-embedding'

DEFAULT_DATA_YAML_FILEPATH = os.path.join(os.path.sep, dir_path, 'config', "time_series_data_config.yml")
TEST_DATA_YAML_FILEPATH = os.path.join(os.path.sep, dir_path, 'config', "test_time_series_data_config.yml")
DEFAULT_EMBEDDER_YAML_FILENAME = os.path.join(os.path.sep, dir_path, 'config', "time_series_embedder_config.yml")
TEST_EMBEDDER_YAML_FILENAME = os.path.join(os.path.sep, dir_path, 'config', "test_time_series_embedder_config.yml")
