"""Pilar 2 — Seleção de Tools Relevantes.

O catálogo de tools está em `data/tools_registry.json`. Passar todas as tools no prompt
de um LLM não escala (estoura contexto, confunde o modelo, aumenta custo e latência).

`search(query, k=2)` deve retornar as `k` tools mais relevantes do catálogo para a query,
antes de qualquer chamada ao LLM. A estratégia de seleção/ranking é livre — escolha o que
fizer sentido e esteja preparado para justificar os trade-offs.
"""
import time
from typing import List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from candidate_starter.embeddings import E5ToolEmbedder
from common.interfaces import BaseToolRetriever
from common.schemas import RetrievalResult, Tool, ToolMatch


class ToolRetriever(BaseToolRetriever):
    def __init__(self) -> None:
        self._tools: List[Tool] = []
        self._tool_embeddings: Optional[np.ndarray] = None
        self._fitted = False
        self.embedder = E5ToolEmbedder()

    def fit(self, tools: List[Tool]) -> "ToolRetriever":
        """Indexa o catálogo de tools para busca."""
        if not tools:
            raise ValueError("Informe pelo menos uma tool para indexar.")

        self._tools = list(tools)
        tool_texts = [self._tool_to_text(tool) for tool in self._tools]

        # A matriz é calculada uma única vez no fit e reutilizada em toda busca.
        # Seu formato é (quantidade de tools, dimensões do embedding).
        self._tool_embeddings = self.embedder.embed_passages(tool_texts)
        self._fitted = True
        return self

    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools mais relevantes para `query`."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de search().")
        if k < 1:
            raise ValueError("k deve ser maior ou igual a 1.")

        start = time.perf_counter()
        # O E5 usa representação assimétrica: a query recebe seu prefixo e
        # gera exatamente um embedding online por chamada de search.
        query_embedding = self.embedder.embed_queries([query])
        scores = cosine_similarity(query_embedding, self._tool_embeddings)[0]

        result_count = min(k, len(self._tools))
        top_indices = np.argsort(-scores, kind="stable")[:result_count]
        matches = [
            ToolMatch(name=self._tools[index].name, score=float(scores[index]))
            for index in top_indices
        ]
        latency_ms = (time.perf_counter() - start) * 1_000

        return RetrievalResult(matches=matches, latency_ms=latency_ms)

    @staticmethod
    def _tool_to_text(tool: Tool) -> str:
        readable_name = tool.name.replace("_", " ")
        return f"{readable_name}. {tool.description} Categoria: {tool.category}."
