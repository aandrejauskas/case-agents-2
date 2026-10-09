"""Estruturas de dados compartilhadas entre o starter e a solução de referência."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Tool:
    name: str
    description: str
    category: str


@dataclass
class RouteResult:
    route: str  # "FAST_PATH" ou "AGENT"
    latency_ms: float
    confidence: Optional[float] = None


@dataclass
class ToolMatch:
    name: str
    score: float


@dataclass
class RetrievalResult:
    matches: List[ToolMatch]
    latency_ms: float
