"""Adaptadores de embeddings do Retriever e dos experimentos diagnósticos."""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_SMALL_MODEL_NAME = "intfloat/multilingual-e5-small"


class TextEmbedder:
    """Embedder MiniLM preservado para reprodução dos experimentos."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Retorna uma matriz (número de textos, dimensão do embedding)."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class E5ToolEmbedder:
    """Embedder E5 com representações assimétricas para query e tool."""

    def __init__(self, model_name: str = E5_SMALL_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """Gera embeddings normalizados com o prefixo de query do E5."""
        return self._embed([f"query: {text}" for text in texts])

    def embed_passages(self, texts: List[str]) -> np.ndarray:
        """Gera embeddings normalizados com o prefixo de passage do E5."""
        return self._embed([f"passage: {text}" for text in texts])

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
