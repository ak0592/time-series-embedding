import os

dir_path = '/home/lab/akira/research-embedding/notebooks'
# for docker
# dir_path = '/workspace/notebooks'
DEFAULT_DATA_YAML_FILEPATH = os.path.join(os.path.sep, dir_path, 'time_series_embedding', 'config', "time_series_data_config.yml")
TEST_DATA_YAML_FILEPATH = os.path.join(os.path.sep, dir_path, 'time_series_embedding', 'config', "test_time_series_data_config.yml")
DEFAULT_EMBEDDER_YAML_FILENAME = os.path.join(os.path.sep, dir_path, 'time_series_embedding', 'config', "time_series_embedder_config.yml")
TEST_EMBEDDER_YAML_FILENAME = os.path.join(os.path.sep, dir_path, 'time_series_embedding', 'config', "test_time_series_embedder_config.yml")
