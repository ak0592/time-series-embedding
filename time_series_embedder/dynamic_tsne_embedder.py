import torch

from time_series_embedder.parent_embedder import ParentTimeSeriesEmbedder
from config.time_series_config import param_to_data_key
from time_series_embedder.related_functions.dynamic_t_sne_functions import dynamic_tsne
from typing import Union


class DynamicTSNEEmbedder(ParentTimeSeriesEmbedder):

    def __init__(self, *args) -> None:
        super(DynamicTSNEEmbedder, self).__init__(*args)
        self.class_key = param_to_data_key('DYNAMIC_T_SNE', self.params)

    def exec_embed(self, embed_dim: int = 2, device: Union[torch.device, str] = 'cpu') -> None:
        self.coordinates = dynamic_tsne(self.df, output_dims=embed_dim, device=device, **self.params)
