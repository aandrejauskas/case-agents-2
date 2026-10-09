"""Compara três candidatos finais de Retriever no fluxo end-to-end do case."""
import gc
from statistics import median
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.evaluate_candidate_union import (
    RRF_C,
    TFIDF_OPTIONS,
    TOP_K_PER_RETRIEVER,
    calculate_metrics,
    current_field_rerank,
    expected_local_rank,
    rank_positions,
    tool_to_text,
)
from candidate_starter.harness import compute_router_metrics, compute_savings
from candidate_starter.router import QueryRouter
from common.data_loader import (
    load_eval_dataset,
    load_router_training_data,
    load_tools,
)
from common.mock_llm import (
    COST_AGENT_LLM_CALL_USD,
    COST_BASELINE_LLM_CALL_USD,
    COST_RETRIEVAL_USD,
    COST_ROUTER_USD,
)


E5_SMALL_MODEL_ID = "intfloat/multilingual-e5-small"
E5_BASE_MODEL_ID = "intfloat/multilingual-e5-base"
STRATEGY_NAMES = [
    "E5_SMALL_DENSE",
    "E5_BASE_DENSE",
    "E5_SMALL_HYBRID_FIELD",
]
METRIC_NAMES = ["Hit@1", "Hit@2", "Hit@3", "Hit@5", "Hit@10", "MRR"]


def percentile_95(values: List[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def latency_summary(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "median": float(median(values)) if values else 0.0,
        "p95": percentile_95(values),
    }


def format_rank(rank: Optional[int]) -> str:
    return "fora do Top-10" if rank is None else str(rank)


def initialize_e5(model_id: str, tool_texts: List[str]):
    initialization_start = perf_counter()
    model = SentenceTransformer(model_id, device="cpu")
    initialization_latency_ms = (perf_counter() - initialization_start) * 1_000

    tool_embedding_start = perf_counter()
    tool_embeddings = model.encode(
        [f"passage: {text}" for text in tool_texts],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    tool_embedding_latency_ms = (perf_counter() - tool_embedding_start) * 1_000
    return model, tool_embeddings, {
        "model_id": model_id,
        "device": str(model.device),
        "dimension": int(tool_embeddings.shape[1]),
        "initialization_latency_ms": initialization_latency_ms,
        "tool_embedding_latency_ms": tool_embedding_latency_ms,
    }


def dense_query_result(
    model: SentenceTransformer,
    tool_embeddings: np.ndarray,
    query: str,
) -> dict:
    embedding_start = perf_counter()
    query_embedding = model.encode(
        [f"query: {query}"],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embedding_latency_ms = (perf_counter() - embedding_start) * 1_000

    ranking_start = perf_counter()
    scores = cosine_similarity(query_embedding, tool_embeddings)[0]
    order = np.argsort(-scores, kind="stable")
    positions = rank_positions(order)
    ranking_latency_ms = (perf_counter() - ranking_start) * 1_000
    return {
        "scores": scores,
        "order": order,
        "positions": positions,
        "query_embedding_latency_ms": query_embedding_latency_ms,
        "dense_ranking_latency_ms": ranking_latency_ms,
    }


def evaluate_e5_small(
    eval_dataset: List[dict],
    tool_names: List[str],
    tool_index_by_name: Dict[str, int],
    tool_texts: List[str],
    combined_vectorizer: TfidfVectorizer,
    combined_tool_matrix,
    name_vectorizer: TfidfVectorizer,
    name_tool_matrix,
    description_vectorizer: TfidfVectorizer,
    description_tool_matrix,
) -> Dict[str, dict]:
    model, tool_embeddings, model_info = initialize_e5(
        E5_SMALL_MODEL_ID, tool_texts
    )
    dense_results = []
    hybrid_results = []

    for item in eval_dataset:
        query = item["query"]
        expected_tool = item.get("expected_tool")
        expected_index = (
            tool_index_by_name[expected_tool] if expected_tool else None
        )
        dense = dense_query_result(model, tool_embeddings, query)

        dense_rank = (
            int(dense["positions"][expected_index])
            if expected_index is not None
            else None
        )
        dense_total_latency_ms = (
            dense["query_embedding_latency_ms"]
            + dense["dense_ranking_latency_ms"]
        )
        dense_results.append(
            {
                "rank": dense_rank,
                "top_2": [tool_names[index] for index in dense["order"][:2]],
                "query_embedding_latency_ms": dense["query_embedding_latency_ms"],
                "dense_ranking_latency_ms": dense["dense_ranking_latency_ms"],
                "extra_latency_ms": 0.0,
                "retriever_latency_ms": dense_total_latency_ms,
            }
        )

        extra_start = perf_counter()
        tfidf_query = combined_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(tfidf_query, combined_tool_matrix)[0]
        tfidf_order = np.argsort(-tfidf_scores, kind="stable")
        tfidf_positions = rank_positions(tfidf_order)
        hybrid_scores = (
            1 / (RRF_C + dense["positions"])
            + 1 / (RRF_C + tfidf_positions)
        )
        hybrid_order = np.argsort(-hybrid_scores, kind="stable")
        hybrid_candidates = hybrid_order[:TOP_K_PER_RETRIEVER]
        field_order, field_positions = current_field_rerank(
            query,
            hybrid_candidates,
            name_vectorizer,
            name_tool_matrix,
            description_vectorizer,
            description_tool_matrix,
        )
        extra_latency_ms = (perf_counter() - extra_start) * 1_000
        final_order = hybrid_candidates[field_order]
        hybrid_rank = (
            expected_local_rank(
                hybrid_candidates, field_positions, expected_index
            )
            if expected_index is not None
            else None
        )
        hybrid_results.append(
            {
                "rank": hybrid_rank,
                "top_2": [tool_names[index] for index in final_order[:2]],
                "query_embedding_latency_ms": dense["query_embedding_latency_ms"],
                "dense_ranking_latency_ms": dense["dense_ranking_latency_ms"],
                "extra_latency_ms": extra_latency_ms,
                "retriever_latency_ms": (
                    dense_total_latency_ms + extra_latency_ms
                ),
            }
        )

    del tool_embeddings
    del model
    gc.collect()
    return {
        "E5_SMALL_DENSE": {
            "model_info": model_info,
            "results": dense_results,
        },
        "E5_SMALL_HYBRID_FIELD": {
            "model_info": model_info,
            "results": hybrid_results,
        },
    }


def evaluate_e5_base(
    eval_dataset: List[dict],
    tool_names: List[str],
    tool_index_by_name: Dict[str, int],
    tool_texts: List[str],
) -> dict:
    model, tool_embeddings, model_info = initialize_e5(
        E5_BASE_MODEL_ID, tool_texts
    )
    results = []
    for item in eval_dataset:
        expected_tool = item.get("expected_tool")
        expected_index = (
            tool_index_by_name[expected_tool] if expected_tool else None
        )
        dense = dense_query_result(model, tool_embeddings, item["query"])
        rank = (
            int(dense["positions"][expected_index])
            if expected_index is not None
            else None
        )
        results.append(
            {
                "rank": rank,
                "top_2": [tool_names[index] for index in dense["order"][:2]],
                "query_embedding_latency_ms": dense["query_embedding_latency_ms"],
                "dense_ranking_latency_ms": dense["dense_ranking_latency_ms"],
                "extra_latency_ms": 0.0,
                "retriever_latency_ms": (
                    dense["query_embedding_latency_ms"]
                    + dense["dense_ranking_latency_ms"]
                ),
            }
        )

    del tool_embeddings
    del model
    gc.collect()
    return {"model_info": model_info, "results": results}


def isolated_metrics(
    strategy: dict, eval_dataset: List[dict], tool_count: int
) -> Dict[str, float]:
    ranks = [
        result["rank"]
        for item, result in zip(eval_dataset, strategy["results"])
        if item.get("expected_tool")
    ]
    if all(rank is not None for rank in ranks):
        missing_ranks = [tool_count + 1] * len(ranks)
    else:
        missing_ranks = [TOP_K_PER_RETRIEVER + 1] * len(ranks)
    return calculate_metrics(ranks, missing_ranks)


def end_to_end_metrics(
    strategy: dict,
    eval_dataset: List[dict],
    router_results: List[dict],
) -> dict:
    required_tool_total = sum(bool(item.get("expected_tool")) for item in eval_dataset)
    reached_indices = [
        index
        for index, (item, route) in enumerate(zip(eval_dataset, router_results))
        if item.get("expected_tool") and route["route"] == "AGENT"
    ]
    reached_hits = sum(
        strategy["results"][index]["rank"] is not None
        and strategy["results"][index]["rank"] <= 2
        for index in reached_indices
    )
    end_to_end_hits = sum(
        bool(item.get("expected_tool"))
        and router_results[index]["route"] == "AGENT"
        and strategy["results"][index]["rank"] is not None
        and strategy["results"][index]["rank"] <= 2
        for index, item in enumerate(eval_dataset)
    )
    return {
        "reached_retriever": len(reached_indices),
        "reached_hit_at_2": (
            reached_hits / len(reached_indices) if reached_indices else 0.0
        ),
        "reached_hits": reached_hits,
        "end_to_end_hit_at_2": (
            end_to_end_hits / required_tool_total
            if required_tool_total
            else 0.0
        ),
        "end_to_end_hits": end_to_end_hits,
        "required_tool_total": required_tool_total,
    }


def compute_costs(query_count: int, agent_routes: int) -> dict:
    smart_cost = (
        query_count * COST_ROUTER_USD
        + agent_routes * COST_RETRIEVAL_USD
        + agent_routes * COST_AGENT_LLM_CALL_USD
    )
    baseline_cost = query_count * COST_BASELINE_LLM_CALL_USD
    savings = compute_savings(smart_cost, 0.0, baseline_cost, 0.0)
    return {
        "smart_cost": smart_cost,
        "baseline_cost": baseline_cost,
        "cost_savings_pct": savings["cost_savings_pct"],
    }


def main() -> None:
    tools = load_tools()
    eval_dataset = load_eval_dataset()
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}
    tool_texts = [tool_to_text(tool) for tool in tools]

    train_texts, train_labels = load_router_training_data()
    router = QueryRouter().fit(train_texts, train_labels)
    router_results = []
    for item in eval_dataset:
        result = router.predict(item["query"])
        router_results.append(
            {
                "route": result.route,
                "latency_ms": result.latency_ms,
                "confidence": result.confidence,
            }
        )

    combined_vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    combined_tool_matrix = combined_vectorizer.fit_transform(tool_texts)
    name_vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    name_tool_matrix = name_vectorizer.fit_transform(
        [tool.name.replace("_", " ") for tool in tools]
    )
    description_vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    description_tool_matrix = description_vectorizer.fit_transform(
        [tool.description for tool in tools]
    )

    print("Executando E5 Small...", flush=True)
    strategies = evaluate_e5_small(
        eval_dataset,
        tool_names,
        tool_index_by_name,
        tool_texts,
        combined_vectorizer,
        combined_tool_matrix,
        name_vectorizer,
        name_tool_matrix,
        description_vectorizer,
        description_tool_matrix,
    )
    print("Executando E5 Base...", flush=True)
    strategies["E5_BASE_DENSE"] = evaluate_e5_base(
        eval_dataset,
        tool_names,
        tool_index_by_name,
        tool_texts,
    )

    for strategy in strategies.values():
        strategy["isolated_metrics"] = isolated_metrics(
            strategy, eval_dataset, len(tools)
        )
        strategy["end_to_end_metrics"] = end_to_end_metrics(
            strategy, eval_dataset, router_results
        )

    actual_routes = [item["expected_route"] for item in eval_dataset]
    predicted_routes = [result["route"] for result in router_results]
    router_metrics = compute_router_metrics(
        actual_routes, predicted_routes, ["FAST_PATH", "AGENT"]
    )
    agent_routes = sum(route == "AGENT" for route in predicted_routes)
    costs = compute_costs(len(eval_dataset), agent_routes)
    print_report(
        eval_dataset,
        router_results,
        router_metrics,
        strategies,
        agent_routes,
        costs,
    )


def print_metric_table(strategies: Dict[str, dict]) -> None:
    print(f"{'Métrica':<16}" + "".join(f"{name:>27}" for name in STRATEGY_NAMES))
    for metric_name in METRIC_NAMES:
        print(
            f"{metric_name:<16}"
            + "".join(
                f"{strategies[name]['isolated_metrics'][metric_name]:>27.6f}"
                for name in STRATEGY_NAMES
            )
        )
    print(
        f"{'Mediana rank':<16}"
        + "".join(
            f"{strategies[name]['isolated_metrics']['median_rank']:>27.1f}"
            for name in STRATEGY_NAMES
        )
    )


def print_latency(label: str, values: List[float]) -> None:
    stats = latency_summary(values)
    print(
        f"   {label:<26} média={stats['mean']:.3f} ms | "
        f"mediana={stats['median']:.3f} ms | p95={stats['p95']:.3f} ms"
    )


def comparison_lists(strategies: Dict[str, dict], eval_dataset: List[dict]):
    expected_indices = [
        index for index, item in enumerate(eval_dataset) if item.get("expected_tool")
    ]

    def better(left_name: str, right_name: str):
        rows = []
        for index in expected_indices:
            left = strategies[left_name]["results"][index]["rank"]
            right = strategies[right_name]["results"][index]["rank"]
            left_value = float("inf") if left is None else left
            right_value = float("inf") if right is None else right
            if left_value < right_value:
                rows.append(index)
        return rows

    return {
        "small_over_hybrid": better(
            "E5_SMALL_DENSE", "E5_SMALL_HYBRID_FIELD"
        ),
        "hybrid_over_small": better(
            "E5_SMALL_HYBRID_FIELD", "E5_SMALL_DENSE"
        ),
        "base_over_small": better("E5_BASE_DENSE", "E5_SMALL_DENSE"),
        "small_over_base": better("E5_SMALL_DENSE", "E5_BASE_DENSE"),
    }


def print_report(
    eval_dataset: List[dict],
    router_results: List[dict],
    router_metrics: dict,
    strategies: Dict[str, dict],
    agent_routes: int,
    costs: dict,
) -> None:
    print()
    print("CANDIDATOS FINAIS DE RETRIEVER — COMPARAÇÃO END-TO-END")
    print("=" * 110)
    print(
        "AVISO: o eval_dataset já foi usado em vários experimentos. "
        "Os resultados são diagnósticos/comparativos, não um holdout cego."
    )
    print()

    print("ROUTER REUTILIZADO NAS TRÊS ALTERNATIVAS")
    print("=" * 110)
    print(f"Accuracy: {router_metrics['accuracy']:.2%}")
    print(f"Confusion matrix: {router_metrics['confusion_matrix']}")
    print(f"Queries roteadas como AGENT: {agent_routes}/{len(eval_dataset)}")
    print_latency(
        "Router",
        [result["latency_ms"] for result in router_results],
    )
    print()

    print("RETRIEVER ISOLADO")
    print("=" * 110)
    print_metric_table(strategies)
    print()

    print("PIPELINE END-TO-END")
    print("=" * 110)
    for name in STRATEGY_NAMES:
        metrics = strategies[name]["end_to_end_metrics"]
        print(
            f"{name}: chegaram ao Retriever={metrics['reached_retriever']}; "
            f"Hit@2 entre as que chegaram="
            f"{metrics['reached_hit_at_2']:.2%} "
            f"({metrics['reached_hits']}/{metrics['reached_retriever']}); "
            f"sucesso end-to-end={metrics['end_to_end_hit_at_2']:.2%} "
            f"({metrics['end_to_end_hits']}/{metrics['required_tool_total']})"
        )
    print(
        "Retriever Hit@2 isolado usa todas as 20 queries com expected_tool. "
        "O sucesso end-to-end mantém essas 20 no denominador e penaliza uma "
        "query quando o Router a envia incorretamente ao FAST_PATH."
    )
    print()

    print("COMPARAÇÃO POR QUERY")
    print("=" * 110)
    for index, item in enumerate(eval_dataset):
        if not item.get("expected_tool"):
            continue
        print(f"Query: {item['query']}")
        print(f"   expected_tool: {item['expected_tool']}")
        print(f"   Router: {router_results[index]['route']}")
        for name in STRATEGY_NAMES:
            result = strategies[name]["results"][index]
            print(
                f"   {name}: rank={format_rank(result['rank'])}; "
                f"Top-2={result['top_2']}"
            )
        print()

    print("LATÊNCIA OFFLINE E ONLINE EM CPU")
    print("=" * 110)
    agent_indices = [
        index
        for index, result in enumerate(router_results)
        if result["route"] == "AGENT"
    ]
    print(
        "Componentes do Retriever: apenas as 21 queries AGENT. "
        "Router + Retriever: todas as 30 queries, com apenas Router no FAST_PATH."
    )
    for name in STRATEGY_NAMES:
        strategy = strategies[name]
        info = strategy["model_info"]
        results = [strategy["results"][index] for index in agent_indices]
        pipeline_latencies = [
            router_results[index]["latency_ms"]
            + (
                strategy["results"][index]["retriever_latency_ms"]
                if router_results[index]["route"] == "AGENT"
                else 0.0
            )
            for index in range(len(eval_dataset))
        ]
        print(f"{name}:")
        print(
            f"   model={info['model_id']} | device={info['device']} | "
            f"dimensão={info['dimension']}"
        )
        print(
            f"   inicialização={info['initialization_latency_ms']:.3f} ms | "
            f"embeddings das 285 tools={info['tool_embedding_latency_ms']:.3f} ms"
        )
        print_latency(
            "Embedding da query",
            [result["query_embedding_latency_ms"] for result in results],
        )
        print_latency(
            "Ranking Dense",
            [result["dense_ranking_latency_ms"] for result in results],
        )
        print_latency(
            "Etapas extras",
            [result["extra_latency_ms"] for result in results],
        )
        print_latency(
            "Retriever total",
            [result["retriever_latency_ms"] for result in results],
        )
        print_latency("Router + Retriever", pipeline_latencies)
        print()

    print("CUSTO MONETÁRIO — IDÊNTICO NAS TRÊS ALTERNATIVAS")
    print("=" * 110)
    print(f"Pipeline inteligente: ${costs['smart_cost']:.6f}")
    print(f"Baseline com LLM caro: ${costs['baseline_cost']:.6f}")
    print(f"Economia: {costs['cost_savings_pct']:.4f}%")
    print(
        "Os três Retrievers são locais e antecedem o mesmo agente; portanto "
        "a troca entre eles não altera o custo monetário definido pelo harness."
    )
    print()

    print("CUSTO COMPUTACIONAL E COMPLEXIDADE")
    print("=" * 110)
    print(
        "E5_SMALL_DENSE: 1 embedding/query de 384 dimensões; 285 cosine "
        "similarities; SentenceTransformer + cosine + ordenação exata."
    )
    print(
        "E5_BASE_DENSE: 1 embedding/query de 768 dimensões; 285 cosine "
        "similarities; mesmas etapas do Small, com modelo e vetores maiores."
    )
    print(
        "E5_SMALL_HYBRID_FIELD: 1 embedding/query de 384 dimensões; 285 "
        "cosines Dense + 285 TF-IDF + 10 name + 10 description; acrescenta "
        "três índices TF-IDF, Hybrid RRF, Top-10 e Field-Aware RRF."
    )
    print()

    comparisons = comparison_lists(strategies, eval_dataset)
    print("VENCEDORES POR QUERY")
    print("=" * 110)
    labels = [
        ("Small Dense supera Hybrid + Field", "small_over_hybrid"),
        ("Hybrid + Field supera Small Dense", "hybrid_over_small"),
        ("Base Dense supera Small Dense", "base_over_small"),
        ("Small Dense supera Base Dense", "small_over_base"),
    ]
    for label, key in labels:
        print(f"{label}:")
        if comparisons[key]:
            for index in comparisons[key]:
                print(
                    f"- {eval_dataset[index]['query']} | "
                    f"{eval_dataset[index]['expected_tool']}"
                )
        else:
            print("- nenhuma")


if __name__ == "__main__":
    main()
