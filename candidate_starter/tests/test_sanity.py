"""Testes de sanidade que o candidato deve fazer passar.

Antes de implementar router.py/retrieval.py, esses testes falham com
NotImplementedError — isso é esperado, servem de guia (estilo TDD).
"""
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter
from common.data_loader import load_tools

TRAIN_TEXTS = [
    "Bom dia, qual o horário de atendimento?",
    "Oi, tudo bem?",
    "Quero saber meu saldo",
    "Preciso bloquear meu cartão perdido",
]
TRAIN_LABELS = ["FAST_PATH", "FAST_PATH", "AGENT", "AGENT"]


def test_router_predicts_valid_route():
    router = QueryRouter().fit(TRAIN_TEXTS, TRAIN_LABELS)
    result = router.predict("Bom dia!")
    assert result.route in {"FAST_PATH", "AGENT"}
    assert result.latency_ms >= 0


def test_retriever_returns_k_matches():
    tools = load_tools()
    retriever = ToolRetriever().fit(tools)
    result = retriever.search("Quero saber meu saldo", k=2)
    assert len(result.matches) == 2
    assert all(m.name for m in result.matches)
