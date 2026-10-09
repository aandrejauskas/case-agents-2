"""Compara modelos de embedding no Dense puro e no pipeline Field-Aware."""
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
from common.data_loader import load_eval_dataset, load_tools


MODEL_CONFIGS = [
    {
        "name": "CURRENT_MINILM",
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "e5_prefixes": False,
        "normalize_embeddings": False,
    },
    {
        "name": "E5_SMALL",
        "model_id": "intfloat/multilingual-e5-small",
        "e5_prefixes": True,
        "normalize_embeddings": True,
    },
    {
        "name": "E5_BASE",
        "model_id": "intfloat/multilingual-e5-base",
        "e5_prefixes": True,
        "normalize_embeddings": True,
    },
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


def query_text(query: str, uses_e5_prefixes: bool) -> str:
    return f"query: {query}" if uses_e5_prefixes else query


def tool_texts_for_model(
    tool_texts: List[str], uses_e5_prefixes: bool
) -> List[str]:
    if uses_e5_prefixes:
        return [f"passage: {text}" for text in tool_texts]
    return tool_texts


def format_rank(rank: Optional[int]) -> str:
    return "fora do Top-10" if rank is None else str(rank)


def run_model_experiment(
    config: dict,
    evaluation_items: List[dict],
    tool_names: List[str],
    tool_index_by_name: Dict[str, int],
    tool_texts: List[str],
    combined_vectorizer: TfidfVectorizer,
    combined_tool_matrix,
    name_vectorizer: TfidfVectorizer,
    name_tool_matrix,
    description_vectorizer: TfidfVectorizer,
    description_tool_matrix,
) -> dict:
    initialization_start = perf_counter()
    model = SentenceTransformer(config["model_id"], device="cpu")
    initialization_latency_ms = (perf_counter() - initialization_start) * 1_000
    device = str(model.device)

    indexed_texts = tool_texts_for_model(
        tool_texts, config["e5_prefixes"]
    )
    tool_embedding_start = perf_counter()
    tool_embeddings = model.encode(
        indexed_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=config["normalize_embeddings"],
    )
    tool_embedding_latency_ms = (perf_counter() - tool_embedding_start) * 1_000

    results: List[dict] = []
    for item in evaluation_items:
        query = item["query"]
        expected_tool = item["expected_tool"]
        expected_index = tool_index_by_name[expected_tool]

        query_embedding_start = perf_counter()
        encoded_query = model.encode(
            [query_text(query, config["e5_prefixes"])],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=config["normalize_embeddings"],
        )
        query_embedding_latency_ms = (
            perf_counter() - query_embedding_start
        ) * 1_000

        dense_post_start = perf_counter()
        dense_scores = cosine_similarity(encoded_query, tool_embeddings)[0]
        dense_order = np.argsort(-dense_scores, kind="stable")
        dense_positions = rank_positions(dense_order)
        dense_post_latency_ms = (perf_counter() - dense_post_start) * 1_000
        dense_latency_ms = query_embedding_latency_ms + dense_post_latency_ms

        pipeline_extra_start = perf_counter()
        tfidf_query = combined_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(tfidf_query, combined_tool_matrix)[0]
        tfidf_order = np.argsort(-tfidf_scores, kind="stable")
        tfidf_positions = rank_positions(tfidf_order)

        hybrid_scores = (
            1 / (RRF_C + dense_positions)
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
        pipeline_extra_latency_ms = (
            perf_counter() - pipeline_extra_start
        ) * 1_000

        results.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "dense_rank": int(dense_positions[expected_index]),
                "pipeline_rank": expected_local_rank(
                    hybrid_candidates, field_positions, expected_index
                ),
                "dense_top_5": [
                    (tool_names[index], float(dense_scores[index]))
                    for index in dense_order[:5]
                ],
                "pipeline_top_5": [
                    tool_names[hybrid_candidates[local_index]]
                    for local_index in field_order[:5]
                ],
                "query_embedding_latency_ms": query_embedding_latency_ms,
                "dense_latency_ms": dense_latency_ms,
                "pipeline_latency_ms": (
                    dense_latency_ms + pipeline_extra_latency_ms
                ),
            }
        )

    full_missing = [len(tool_names) + 1] * len(results)
    field_missing = [TOP_K_PER_RETRIEVER + 1] * len(results)
    output = {
        "config": config,
        "device": device,
        "dimension": int(tool_embeddings.shape[1]),
        "initialization_latency_ms": initialization_latency_ms,
        "tool_embedding_latency_ms": tool_embedding_latency_ms,
        "results": results,
        "dense_metrics": calculate_metrics(
            [result["dense_rank"] for result in results], full_missing
        ),
        "pipeline_metrics": calculate_metrics(
            [result["pipeline_rank"] for result in results], field_missing
        ),
    }

    del tool_embeddings
    del model
    gc.collect()
    return output


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}
    tool_texts = [tool_to_text(tool) for tool in tools]

    # Os índices lexicais são idênticos para os três pipelines e ficam offline.
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

    experiments = {}
    for config in MODEL_CONFIGS:
        print(f"Executando {config['name']}...", flush=True)
        experiments[config["name"]] = run_model_experiment(
            config,
            evaluation_items,
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

    print_report(experiments, evaluation_items)


def print_metrics_table(experiments: dict, metrics_key: str) -> None:
    model_names = [config["name"] for config in MODEL_CONFIGS]
    print(f"{'Métrica':<16}" + "".join(f"{name:>20}" for name in model_names))
    for metric_name in METRIC_NAMES:
        print(
            f"{metric_name:<16}"
            + "".join(
                f"{experiments[name][metrics_key][metric_name]:>20.6f}"
                for name in model_names
            )
        )
    print(
        f"{'Mediana rank':<16}"
        + "".join(
            f"{experiments[name][metrics_key]['median_rank']:>20.1f}"
            for name in model_names
        )
    )


def print_latency_line(label: str, values: List[float]) -> None:
    summary = latency_summary(values)
    print(
        f"   {label:<28} média={summary['mean']:.3f} ms | "
        f"mediana={summary['median']:.3f} ms | p95={summary['p95']:.3f} ms"
    )


def rank_changes(current_results: List[dict], candidate_results: List[dict]):
    recovered = []
    lost = []
    improved = []
    worsened = []
    for current, candidate in zip(current_results, candidate_results):
        old_rank = current["pipeline_rank"]
        new_rank = candidate["pipeline_rank"]
        if old_rank is None and new_rank is not None:
            recovered.append((current, candidate))
        elif old_rank is not None and new_rank is None:
            lost.append((current, candidate))
        elif old_rank is not None and new_rank is not None:
            if new_rank < old_rank:
                improved.append((current, candidate))
            elif new_rank > old_rank:
                worsened.append((current, candidate))
    return recovered, lost, improved, worsened


def print_report(experiments: dict, evaluation_items: List[dict]) -> None:
    model_names = [config["name"] for config in MODEL_CONFIGS]
    print()
    print("COMPARAÇÃO DE EMBEDDING MODELS — DIAGNÓSTICO EXPERIMENTAL")
    print("=" * 106)
    print(f"Queries com expected_tool: {len(evaluation_items)}")
    print("Device forçado para comparação: CPU")
    print(
        "AVISO: o eval_dataset já foi usado nos experimentos anteriores. "
        "Este benchmark é diagnóstico/comparativo e não é um holdout cego."
    )
    print()

    print("RANKS POR QUERY")
    print("=" * 106)
    for index, item in enumerate(evaluation_items, start=1):
        print(f"{index}. Query: {item['query']}")
        print(f"   expected_tool: {item['expected_tool']}")
        print(
            "   Dense: "
            + " | ".join(
                f"{name}={experiments[name]['results'][index - 1]['dense_rank']}"
                for name in model_names
            )
        )
        print(
            "   Pipeline: "
            + " | ".join(
                f"{name}="
                f"{format_rank(experiments[name]['results'][index - 1]['pipeline_rank'])}"
                for name in model_names
            )
        )

        dense_ranks = {
            experiments[name]["results"][index - 1]["dense_rank"]
            for name in model_names
        }
        if len(dense_ranks) > 1:
            print("   Top-5 Dense — diferença relevante entre ranks:")
            for name in model_names:
                top_5 = experiments[name]["results"][index - 1]["dense_top_5"]
                formatted = ", ".join(
                    f"{tool_name} ({score:.4f})" for tool_name, score in top_5
                )
                print(f"      {name}: {formatted}")
        print()

    print("PARTE 1 — DENSE PURO")
    print("=" * 106)
    print_metrics_table(experiments, "dense_metrics")
    print()

    print("PARTE 2 — PIPELINE HYBRID RRF TOP-10 + FIELD-AWARE")
    print("=" * 106)
    print_metrics_table(experiments, "pipeline_metrics")
    print()

    print("LATÊNCIA E CARACTERÍSTICAS")
    print("=" * 106)
    print(
        "Downloads não entram nos números. Inicialização e embeddings das "
        "tools são informados separadamente e ficam fora da latência por query."
    )
    for name in model_names:
        experiment = experiments[name]
        results = experiment["results"]
        print(f"{name}:")
        print(f"   model_id: {experiment['config']['model_id']}")
        print(f"   device: {experiment['device']}")
        print(f"   dimensionalidade: {experiment['dimension']}")
        print(
            "   inicialização/carregamento: "
            f"{experiment['initialization_latency_ms']:.3f} ms"
        )
        print(
            "   embeddings offline das tools: "
            f"{experiment['tool_embedding_latency_ms']:.3f} ms"
        )
        print_latency_line(
            "Embedding da query",
            [result["query_embedding_latency_ms"] for result in results],
        )
        print_latency_line(
            "Dense completo",
            [result["dense_latency_ms"] for result in results],
        )
        print_latency_line(
            "Pipeline final completo",
            [result["pipeline_latency_ms"] for result in results],
        )
        print()

    current_results = experiments["CURRENT_MINILM"]["results"]
    print("MUDANÇAS DO PIPELINE CONTRA CURRENT_MINILM")
    print("=" * 106)
    for name in ["E5_SMALL", "E5_BASE"]:
        candidate_results = experiments[name]["results"]
        recovered, lost, improved, worsened = rank_changes(
            current_results, candidate_results
        )
        new_top_2 = [
            (current, candidate)
            for current, candidate in zip(current_results, candidate_results)
            if (
                current["pipeline_rank"] is None
                or current["pipeline_rank"] > 2
            )
            and candidate["pipeline_rank"] is not None
            and candidate["pipeline_rank"] <= 2
        ]
        lost_top_2 = [
            (current, candidate)
            for current, candidate in zip(current_results, candidate_results)
            if current["pipeline_rank"] is not None
            and current["pipeline_rank"] <= 2
            and (
                candidate["pipeline_rank"] is None
                or candidate["pipeline_rank"] > 2
            )
        ]

        print(f"{name}:")
        print(
            f"   novas no Top-2={len(new_top_2)} | "
            f"saíram do Top-2={len(lost_top_2)} | "
            f"saldo={len(new_top_2) - len(lost_top_2):+d}"
        )
        print("   Recuperadas:")
        if recovered:
            for current, candidate in recovered:
                print(
                    f"   - {current['query']} | {current['expected_tool']}: "
                    f"fora -> {candidate['pipeline_rank']}"
                )
        else:
            print("   - nenhuma")
        print("   Melhoraram:")
        if improved:
            for current, candidate in improved:
                print(
                    f"   - {current['query']} | {current['expected_tool']}: "
                    f"{current['pipeline_rank']} -> {candidate['pipeline_rank']}"
                )
        else:
            print("   - nenhuma")
        print("   Pioraram:")
        if worsened:
            for current, candidate in worsened:
                print(
                    f"   - {current['query']} | {current['expected_tool']}: "
                    f"{current['pipeline_rank']} -> {candidate['pipeline_rank']}"
                )
        else:
            print("   - nenhuma")
        print("   Perdidas do Top-10:")
        if lost:
            for current, candidate in lost:
                print(f"   - {current['query']} | {current['expected_tool']}")
        else:
            print("   - nenhuma")
        print()


if __name__ == "__main__":
    main()
