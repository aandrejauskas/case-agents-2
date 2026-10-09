"""Testes específicos da busca vetorial de tools."""
import pytest

from candidate_starter.retrieval import ToolRetriever
from common.data_loader import load_tools


@pytest.fixture(scope="module")
def fitted_retriever() -> ToolRetriever:
    return ToolRetriever().fit(load_tools())


def test_search_returns_at_most_k_ranked_matches(
    fitted_retriever: ToolRetriever,
) -> None:
    result = fitted_retriever.search("Quero consultar meu saldo", k=2)

    assert len(result.matches) <= 2
    assert all(match.name for match in result.matches)
    assert all(isinstance(match.score, float) for match in result.matches)
    assert [match.score for match in result.matches] == sorted(
        (match.score for match in result.matches), reverse=True
    )
    assert result.latency_ms >= 0


def test_search_finds_balance_tool(fitted_retriever: ToolRetriever) -> None:
    result = fitted_retriever.search(
        "Quero saber quanto dinheiro tenho disponível na minha conta corrente",
        k=5,
    )

    assert any("saldo" in match.name for match in result.matches)


def test_search_finds_card_blocking_tool(fitted_retriever: ToolRetriever) -> None:
    result = fitted_retriever.search(
        "Perdi meu cartão na rua e preciso bloquear agora",
        k=5,
    )

    assert any(
        "bloqu" in match.name and "cartao" in match.name
        for match in result.matches
    )


def test_search_rejects_k_smaller_than_one(
    fitted_retriever: ToolRetriever,
) -> None:
    with pytest.raises(ValueError, match="k deve ser maior ou igual a 1"):
        fitted_retriever.search("Consultar saldo", k=0)


def test_search_caps_k_at_catalog_size(fitted_retriever: ToolRetriever) -> None:
    tools = load_tools()
    result = fitted_retriever.search("Consultar saldo", k=len(tools) + 10)

    assert len(result.matches) == len(tools)


def test_search_requires_fit() -> None:
    with pytest.raises(RuntimeError, match=r"Chame fit\(\) antes de search"):
        ToolRetriever().search("Consultar saldo")


def test_fit_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="pelo menos uma tool"):
        ToolRetriever().fit([])
