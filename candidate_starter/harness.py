"""Evaluation Harness do Router, Retriever e baseline simulado.

Calcula qualidade, custo e latência do pipeline inteligente e mantém o formato
de relatório exigido pelo case.
"""
import time
from typing import Dict, List

from common.interfaces import BaseRouter, BaseToolRetriever
from common.mock_llm import (
    COST_RETRIEVAL_USD,
    COST_ROUTER_USD,
    fast_path_answer,
    mock_tool_execution,
    simulate_agent_llm_call,
    simulate_baseline_llm_call,
)
from common.schemas import Tool


def compute_router_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """Calcula a acurácia e a matriz de confusão do router.

    Deve retornar um dict no formato:
        {
            "accuracy": 0.9,
            "confusion_matrix": {
                "FAST_PATH": {"FAST_PATH": 5, "AGENT": 1},
                "AGENT": {"FAST_PATH": 0, "AGENT": 10},
            },
        }
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter a mesma quantidade de itens.")

    confusion_matrix = {
        actual: {predicted: 0 for predicted in labels} for actual in labels
    }
    correct = 0

    for actual, predicted in zip(y_true, y_pred):
        confusion_matrix[actual][predicted] += 1
        correct += int(actual == predicted)

    accuracy = correct / len(y_true) if y_true else 0.0
    return {"accuracy": accuracy, "confusion_matrix": confusion_matrix}


def compute_precision_at_k(hits: List[int]) -> float:
    """Calcula a média dos hits binários produzidos pelo Retriever.

    O case chama a métrica de Precision@K, mas, com uma única tool esperada por
    query, ela equivale conceitualmente a Hit Rate@K / Recall@K.
    """
    return sum(hits) / len(hits) if hits else 0.0


def compute_savings(
    smart_cost_usd: float,
    smart_latency_ms: float,
    baseline_cost_usd: float,
    baseline_latency_ms: float,
) -> Dict:
    """Calcula a % de economia de custo e latência do pipeline inteligente em
    relação ao baseline (mandar tudo pro LLM caro).

    Deve retornar um dict no formato:
        {"cost_savings_pct": 65.0, "latency_savings_pct": 40.0}
    """
    cost_savings_pct = (
        (baseline_cost_usd - smart_cost_usd) / baseline_cost_usd * 100
        if baseline_cost_usd
        else 0.0
    )
    latency_savings_pct = (
        (baseline_latency_ms - smart_latency_ms) / baseline_latency_ms * 100
        if baseline_latency_ms
        else 0.0
    )
    return {
        "cost_savings_pct": cost_savings_pct,
        "latency_savings_pct": latency_savings_pct,
    }


def run_harness(
    router: BaseRouter,
    retriever: BaseToolRetriever,
    tools: List[Tool],
    eval_dataset: List[dict],
    k: int = 2,
) -> dict:
    labels = ["FAST_PATH", "AGENT"]

    y_true: List[str] = []
    y_pred: List[str] = []
    precision_hits: List[int] = []

    smart_cost_total = 0.0
    smart_latency_ms_total = 0.0
    baseline_cost_total = 0.0
    baseline_latency_ms_total = 0.0
    router_latency_ms_total = 0.0
    retriever_latency_ms_total = 0.0
    retriever_query_count = 0
    agent_llm_latency_ms_total = 0.0
    agent_llm_query_count = 0

    rows = []

    for item in eval_dataset:
        query = item["query"]
        expected_route = item["expected_route"]
        expected_tool = item.get("expected_tool")

        route_result = router.predict(query)
        y_true.append(expected_route)
        y_pred.append(route_result.route)
        router_latency_ms_total += route_result.latency_ms

        smart_cost = COST_ROUTER_USD
        smart_latency_ms = route_result.latency_ms

        row = {
            "query": query,
            "expected_route": expected_route,
            "predicted_route": route_result.route,
        }

        if route_result.route == "FAST_PATH":
            fast_path_answer(query)
        else:
            retrieval_result = retriever.search(query, k=k)
            smart_cost += COST_RETRIEVAL_USD
            smart_latency_ms += retrieval_result.latency_ms
            retriever_latency_ms_total += retrieval_result.latency_ms
            retriever_query_count += 1

            top_k_names = [m.name for m in retrieval_result.matches]
            if expected_tool:
                precision_hits.append(int(expected_tool in top_k_names))

            row["retrieved_tools"] = top_k_names
            row["expected_tool"] = expected_tool

            if top_k_names:
                mock_tool_execution(top_k_names[0], query)
                agent_llm_start = time.perf_counter()
                llm_result = simulate_agent_llm_call(query, top_k_names[0])
                agent_llm_latency_ms = (
                    time.perf_counter() - agent_llm_start
                ) * 1_000
                smart_latency_ms += agent_llm_latency_ms
                agent_llm_latency_ms_total += agent_llm_latency_ms
                agent_llm_query_count += 1
                smart_cost += llm_result["cost_usd"]

        smart_cost_total += smart_cost
        smart_latency_ms_total += smart_latency_ms

        baseline_start = time.perf_counter()
        baseline_result = simulate_baseline_llm_call(query)
        baseline_latency_ms_total += (time.perf_counter() - baseline_start) * 1000
        baseline_cost_total += baseline_result["cost_usd"]

        rows.append(row)

    router_metrics = compute_router_metrics(y_true, y_pred, labels)
    precision_at_k = compute_precision_at_k(precision_hits) if precision_hits else None
    savings = compute_savings(
        smart_cost_total, smart_latency_ms_total, baseline_cost_total, baseline_latency_ms_total
    )

    report = {
        "n_queries": len(eval_dataset),
        "router_accuracy": router_metrics["accuracy"],
        "confusion_matrix": router_metrics["confusion_matrix"],
        "precision_at_k": precision_at_k,
        "k": k,
        "smart_pipeline": {"total_cost_usd": smart_cost_total, "total_latency_ms": smart_latency_ms_total},
        "baseline_always_llm": {
            "total_cost_usd": baseline_cost_total,
            "total_latency_ms": baseline_latency_ms_total,
        },
        "latency_breakdown": {
            "router_total_ms": router_latency_ms_total,
            "router_average_ms": (
                router_latency_ms_total / len(eval_dataset)
                if eval_dataset
                else 0.0
            ),
            "retriever_total_ms": retriever_latency_ms_total,
            "retriever_average_ms": (
                retriever_latency_ms_total / retriever_query_count
                if retriever_query_count
                else 0.0
            ),
            "retriever_queries": retriever_query_count,
            "agent_llm_total_ms": agent_llm_latency_ms_total,
            "agent_llm_average_ms": (
                agent_llm_latency_ms_total / agent_llm_query_count
                if agent_llm_query_count
                else 0.0
            ),
            "agent_llm_queries": agent_llm_query_count,
        },
        **savings,
        "rows": rows,
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("HARNESS DE AVALIAÇÃO - Router & Tool Retrieval")
    print("=" * 60)
    print(f"Queries avaliadas: {report['n_queries']}")
    print(f"Acurácia do Router: {report['router_accuracy']:.1%}")
    print(f"Matriz de confusão: {report['confusion_matrix']}")
    if report["precision_at_k"] is not None:
        print(f"Precision@{report['k']} do Retriever: {report['precision_at_k']:.1%}")
    latency_breakdown = report.get("latency_breakdown", {})
    if latency_breakdown:
        print(
            "Latência média do Router: "
            f"{latency_breakdown['router_average_ms']:.1f} ms"
        )
        print(
            "Latência média do Retriever: "
            f"{latency_breakdown['retriever_average_ms']:.1f} ms "
            f"({latency_breakdown['retriever_queries']} queries)"
        )
    print("-" * 60)
    print(f"Custo pipeline inteligente: ${report['smart_pipeline']['total_cost_usd']:.5f}")
    print(f"Custo baseline (tudo pro LLM): ${report['baseline_always_llm']['total_cost_usd']:.5f}")
    print(f"Economia de custo: {report.get('cost_savings_pct', 0):.1f}%")
    print(f"Latência pipeline inteligente: {report['smart_pipeline']['total_latency_ms']:.1f} ms")
    print(f"Latência baseline: {report['baseline_always_llm']['total_latency_ms']:.1f} ms")
    print(f"Economia de latência: {report.get('latency_savings_pct', 0):.1f}%")
    print("=" * 60)
