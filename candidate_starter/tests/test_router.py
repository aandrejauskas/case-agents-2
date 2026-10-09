"""Testes específicos do Router TF-IDF + Logistic Regression."""
import pytest

import candidate_starter.embeddings as embeddings
from candidate_starter.router import QueryRouter
from common.data_loader import load_router_training_data


@pytest.fixture(scope="module")
def trained_router() -> QueryRouter:
    texts, labels = load_router_training_data()
    return QueryRouter().fit(texts, labels)


def test_predict_returns_complete_route_result(trained_router: QueryRouter) -> None:
    result = trained_router.predict("Bom dia!")

    assert result.route in {"FAST_PATH", "AGENT"}
    assert result.latency_ms >= 0
    assert result.confidence is not None
    assert 0 <= result.confidence <= 1


def test_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        QueryRouter().predict("Bom dia!")


def test_router_does_not_instantiate_sentence_transformer(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("O Router não deve instanciar SentenceTransformer.")

    monkeypatch.setattr(embeddings, "SentenceTransformer", fail_if_called)
    QueryRouter()


def test_fit_and_predict_reuse_the_same_vectorizer() -> None:
    texts, labels = load_router_training_data()
    router = QueryRouter()
    original_vectorizer = router.vectorizer

    router.fit(texts, labels)
    vocabulary_after_fit = dict(router.vectorizer.vocabulary_)
    router.predict("Quero consultar meu saldo")

    assert router.vectorizer is original_vectorizer
    assert router.vectorizer.vocabulary_ == vocabulary_after_fit


def test_predicts_agent_for_account_operation(trained_router: QueryRouter) -> None:
    result = trained_router.predict("Quero consultar meu saldo disponível agora")

    assert result.route == "AGENT"


def test_predicts_fast_path_for_greeting_and_faq(trained_router: QueryRouter) -> None:
    result = trained_router.predict("Bom dia, qual é o horário de atendimento?")

    assert result.route == "FAST_PATH"
