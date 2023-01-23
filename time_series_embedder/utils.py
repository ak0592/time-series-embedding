from time_series_embedding.time_series_data import ParentTimeSeriesData
from time_series_embedding.time_series_embedder.parent_embedder import ParentTimeSeriesEmbedder
from time_series_embedding.time_series_embedder.const import TIMESERIESEMBEDDERS_REF
from time_series_embedding.config.time_series_config import TimeSeriesConfig
from time_series_embedding.config.const import DEFAULT_EMBEDDER_YAML_FILENAME, TEST_EMBEDDER_YAML_FILENAME


def choose_time_series_embedder(key: str, data: ParentTimeSeriesData, is_test=False) -> ParentTimeSeriesEmbedder:
    if key not in TIMESERIESEMBEDDERS_REF:
        raise NotImplementedError(f"{key} is not supported")
    embedder_config_file_path = DEFAULT_EMBEDDER_YAML_FILENAME if not is_test else TEST_EMBEDDER_YAML_FILENAME
    embedder_params = TimeSeriesConfig.load_params(embedder_config_file_path)[key]

    return TIMESERIESEMBEDDERS_REF[key](data, embedder_params)
