"""Testes unitários das métricas do harness."""
import pytest

import candidate_starter.harness as harness
from candidate_starter.harness import (
    compute_precision_at_k,
    compute_router_metrics,
    compute_savings,
    run_harness,
)
from common.schemas import RetrievalResult, RouteResult, ToolMatch


LABELS = ["FAST_PATH", "AGENT"]


def test_router_metrics_with_perfect_accuracy() -> None:
    result = compute_router_metrics(
        ["FAST_PATH", "AGENT"],
        ["FAST_PATH", "AGENT"],
        LABELS,
    )

    assert result["accuracy"] == 1.0
    assert result["confusion_matrix"] == {
        "FAST_PATH": {"FAST_PATH": 1, "AGENT": 0},
        "AGENT": {"FAST_PATH": 0, "AGENT": 1},
    }


def test_router_metrics_with_partial_accuracy_and_confusion_matrix() -> None:
    result = compute_router_metrics(
        ["FAST_PATH", "FAST_PATH", "AGENT", "AGENT"],
        ["FAST_PATH", "AGENT", "FAST_PATH", "AGENT"],
        LABELS,
    )

    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"] == {
        "FAST_PATH": {"FAST_PATH": 1, "AGENT": 1},
        "AGENT": {"FAST_PATH": 1, "AGENT": 1},
    }


def test_precision_at_k_with_all_hits() -> None:
    assert compute_precision_at_k([1, 1, 1]) == 1.0


def test_precision_at_k_with_no_hits() -> None:
    assert compute_precision_at_k([0, 0, 0]) == 0.0


def test_savings_with_positive_savings() -> None:
    result = compute_savings(
        smart_cost_usd=25.0,
        smart_latency_ms=60.0,
        baseline_cost_usd=100.0,
        baseline_latency_ms=100.0,
    )

    assert result["cost_savings_pct"] == pytest.approx(75.0)
    assert result["latency_savings_pct"] == pytest.approx(40.0)


def test_savings_with_zero_baselines() -> None:
    result = compute_savings(
        smart_cost_usd=10.0,
        smart_latency_ms=20.0,
        baseline_cost_usd=0.0,
        baseline_latency_ms=0.0,
    )

    assert result == {
        "cost_savings_pct": 0.0,
        "latency_savings_pct": 0.0,
    }


def test_run_harness_counts_agent_llm_latency(monkeypatch) -> None:
    class AgentRouter:
        def predict(self, query: str) -> RouteResult:
            return RouteResult(route="AGENT", latency_ms=2.0, confidence=1.0)

    class Retriever:
        def search(self, query: str, k: int = 2) -> RetrievalResult:
            return RetrievalResult(
                matches=[ToolMatch(name="consultar_saldo", score=0.9)],
                latency_ms=3.0,
            )

    clock = iter([10.0, 10.007, 20.0, 20.011])
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        harness,
        "simulate_agent_llm_call",
        lambda query, tool_name: {"cost_usd": 0.01},
    )
    monkeypatch.setattr(
        harness,
        "simulate_baseline_llm_call",
        lambda query: {"cost_usd": 0.03},
    )

    report = run_harness(
        AgentRouter(),
        Retriever(),
        tools=[],
        eval_dataset=[
            {
                "query": "Quero saber meu saldo",
                "expected_route": "AGENT",
                "expected_tool": "consultar_saldo",
            }
        ],
    )

    # Router (2 ms) + Retriever (3 ms) + Agent LLM medido (7 ms).
    assert report["smart_pipeline"]["total_latency_ms"] == pytest.approx(12.0)
    assert report["baseline_always_llm"]["total_latency_ms"] == pytest.approx(
        11.0
    )
    assert report["latency_breakdown"] == pytest.approx(
        {
            "router_total_ms": 2.0,
            "router_average_ms": 2.0,
            "retriever_total_ms": 3.0,
            "retriever_average_ms": 3.0,
            "retriever_queries": 1,
            "agent_llm_total_ms": 7.0,
            "agent_llm_average_ms": 7.0,
            "agent_llm_queries": 1,
        }
    )
