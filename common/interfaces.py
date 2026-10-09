"""Contratos (ABCs) que tanto o starter quanto a solução de referência implementam.

Isso garante que `run_case.py` funcione de forma idêntica nas duas pastas, trocando
apenas a implementação concreta de Router e Retriever.
"""
from abc import ABC, abstractmethod
from typing import List

from common.schemas import RetrievalResult, RouteResult, Tool


class BaseRouter(ABC):
    """Contrato do Router (Pilar 1)."""

    @abstractmethod
    def fit(self, texts: List[str], labels: List[str]) -> "BaseRouter":
        """Treina o router a partir de exemplos rotulados (FAST_PATH / AGENT)."""

    @abstractmethod
    def predict(self, query: str) -> RouteResult:
        """Classifica uma query e retorna a rota escolhida + latência medida."""


class BaseToolRetriever(ABC):
    """Contrato do componente de Seleção de Tools (Pilar 2)."""

    @abstractmethod
    def fit(self, tools: List[Tool]) -> "BaseToolRetriever":
        """Indexa o catálogo de tools para recuperação."""

    @abstractmethod
    def search(self, query: str, k: int = 2) -> RetrievalResult:
        """Retorna as top-k tools mais relevantes para a query."""
