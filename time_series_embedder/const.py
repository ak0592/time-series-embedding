from time_series_embedding.time_series_embedder.dynamic_tsne_embedder import DynamicTSNEEmbedder
from time_series_embedding.time_series_embedder.isne_embedder import IsneEmbedder


TIMESERIESEMBEDDERS_REF = {
        "DYNAMIC_T_SNE": DynamicTSNEEmbedder,
        "I_SNE": IsneEmbedder
}
