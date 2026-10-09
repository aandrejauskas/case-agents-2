"""Avalia um reranker lexical por campos sobre o Top-10 do Hybrid RRF."""
from statistics import median
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_starter.embeddings import TextEmbedder
from common.data_loader import load_eval_dataset, load_tools
from common.schemas import Tool


RRF_C = 60
CANDIDATE_LIMIT = 10
HIT_CUTOFFS = [1, 2, 3, 5, 10]
TFIDF_OPTIONS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "norm": "l2",
}


def tool_to_text(tool: Tool) -> str:
    """Usa a mesma representação textual dos experimentos anteriores."""
    readable_name = tool.name.replace("_", " ")
    return f"{readable_name}. {tool.description} Categoria: {tool.category}."


def rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
    """Converte uma ordenação de índices na posição de cada item."""
    positions = np.empty(len(sorted_indices), dtype=np.int64)
    positions[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
    return positions


def calculate_metrics(
    ranks: List[Optional[int]], missing_rank: int
) -> Dict[str, float]:
    """Calcula hits, MRR e mediana, atribuindo zero de MRR aos ausentes."""
    total = len(ranks)
    metrics = {
        f"Hit@{cutoff}": (
            sum(rank is not None and rank <= cutoff for rank in ranks) / total
            if total
            else 0.0
        )
        for cutoff in HIT_CUTOFFS
    }
    metrics["MRR"] = (
        sum(0.0 if rank is None else 1 / rank for rank in ranks) / total
        if total
        else 0.0
    )
    median_values = [missing_rank if rank is None else rank for rank in ranks]
    metrics["median_rank"] = float(median(median_values)) if median_values else 0.0
    return metrics


def percentile_95(values: List[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}

    # Índices construídos uma vez. O tempo de fit não entra na latência por query.
    tool_texts = [tool_to_text(tool) for tool in tools]
    embedder = TextEmbedder()
    dense_tool_matrix = embedder.embed(tool_texts)

    combined_vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    combined_tool_matrix = combined_vectorizer.fit_transform(tool_texts)

    name_texts = [tool.name.replace("_", " ") for tool in tools]
    name_vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    name_tool_matrix = name_vectorizer.fit_transform(name_texts)

    description_texts = [tool.description for tool in tools]
    description_vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    description_tool_matrix = description_vectorizer.fit_transform(description_texts)

    results: List[dict] = []
    for item in evaluation_items:
        query = item["query"]
        expected_tool = item["expected_tool"]
        expected_index = tool_index_by_name[expected_tool]

        pipeline_start = perf_counter()

        dense_query = embedder.embed([query])
        dense_scores = cosine_similarity(dense_query, dense_tool_matrix)[0]
        dense_order = np.argsort(-dense_scores, kind="stable")
        dense_positions = rank_positions(dense_order)

        sparse_query = combined_vectorizer.transform([query])
        sparse_scores = cosine_similarity(sparse_query, combined_tool_matrix)[0]
        sparse_order = np.argsort(-sparse_scores, kind="stable")
        sparse_positions = rank_positions(sparse_order)

        hybrid_scores = (
            1 / (RRF_C + dense_positions)
            + 1 / (RRF_C + sparse_positions)
        )
        hybrid_order = np.argsort(-hybrid_scores, kind="stable")
        hybrid_positions = rank_positions(hybrid_order)
        candidate_indices = hybrid_order[:CANDIDATE_LIMIT]

        reranker_start = perf_counter()

        name_query = name_vectorizer.transform([query])
        candidate_name_scores = cosine_similarity(
            name_query, name_tool_matrix[candidate_indices]
        )[0]
        # candidate_indices já está na ordem base. O sort estável mantém essa
        # ordem quando há empates, inclusive quando vários scores são zero.
        name_order = np.argsort(-candidate_name_scores, kind="stable")
        name_positions = rank_positions(name_order)

        description_query = description_vectorizer.transform([query])
        candidate_description_scores = cosine_similarity(
            description_query, description_tool_matrix[candidate_indices]
        )[0]
        description_order = np.argsort(
            -candidate_description_scores, kind="stable"
        )
        description_positions = rank_positions(description_order)

        base_positions = np.arange(1, len(candidate_indices) + 1)
        field_scores = (
            1 / (RRF_C + base_positions)
            + 1 / (RRF_C + name_positions)
            + 1 / (RRF_C + description_positions)
        )
        # Outro sort estável: empate no FIELD_RRF preserva o ranking Hybrid.
        field_order = np.argsort(-field_scores, kind="stable")
        field_positions = rank_positions(field_order)

        reranker_latency_ms = (perf_counter() - reranker_start) * 1_000
        pipeline_latency_ms = (perf_counter() - pipeline_start) * 1_000

        candidate_locations = np.flatnonzero(candidate_indices == expected_index)
        if len(candidate_locations):
            expected_candidate_index = int(candidate_locations[0])
            field_rank: Optional[int] = int(
                field_positions[expected_candidate_index]
            )
            name_rank: Optional[int] = int(
                name_positions[expected_candidate_index]
            )
            description_rank: Optional[int] = int(
                description_positions[expected_candidate_index]
            )
        else:
            field_rank = None
            name_rank = None
            description_rank = None

        results.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "dense_rank": int(dense_positions[expected_index]),
                "tfidf_rank": int(sparse_positions[expected_index]),
                "hybrid_rank": int(hybrid_positions[expected_index]),
                "field_rank": field_rank,
                "name_rank": name_rank,
                "description_rank": description_rank,
                "field_top_5": [
                    (
                        tool_names[candidate_indices[local_index]],
                        float(field_scores[local_index]),
                    )
                    for local_index in field_order[:5]
                ],
                "reranker_latency_ms": reranker_latency_ms,
                "pipeline_latency_ms": pipeline_latency_ms,
            }
        )

    metrics = {
        "DENSE": calculate_metrics(
            [result["dense_rank"] for result in results], len(tools) + 1
        ),
        "TF-IDF": calculate_metrics(
            [result["tfidf_rank"] for result in results], len(tools) + 1
        ),
        "HYBRID RRF": calculate_metrics(
            [result["hybrid_rank"] for result in results], len(tools) + 1
        ),
        "FIELD-AWARE RRF": calculate_metrics(
            [result["field_rank"] for result in results], CANDIDATE_LIMIT + 1
        ),
    }
    print_report(results, metrics)


def format_rank(rank: Optional[int]) -> str:
    return "fora do Top-10" if rank is None else str(rank)


def print_report(results: List[dict], metrics: Dict[str, Dict[str, float]]) -> None:
    reranker_latencies = [result["reranker_latency_ms"] for result in results]
    pipeline_latencies = [result["pipeline_latency_ms"] for result in results]
    outside_candidates = [
        result for result in results if result["field_rank"] is None
    ]

    print("FIELD-AWARE RERANKER — DIAGNÓSTICO EXPERIMENTAL")
    print("=" * 104)
    print(f"Queries com expected_tool: {len(results)}")
    print(f"Geração de candidatos: Dense + TF-IDF + RRF, C={RRF_C}")
    print(f"Candidatos por query: Top-{CANDIDATE_LIMIT}")
    print(
        "Empates lexicais: ordenação estável preserva a ordem anterior do "
        "Hybrid RRF."
    )
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   rank Hybrid RRF: {result['hybrid_rank']}")
        print(f"   rank Field-Aware: {format_rank(result['field_rank'])}")
        print(f"   rank_name nos candidatos: {format_rank(result['name_rank'])}")
        print(
            "   rank_description nos candidatos: "
            f"{format_rank(result['description_rank'])}"
        )
        if result["field_rank"] is None:
            print(
                "   observação: a expected_tool não estava no Top-10 Hybrid; "
                "o reranker não poderia recuperá-la."
            )
        print("   Top-5 Field-Aware:")
        for rank, (name, score) in enumerate(result["field_top_5"], start=1):
            print(f"      {rank}. {name}: {score:.8f}")
        print(f"   latência adicional: {result['reranker_latency_ms']:.3f} ms")
        print(f"   latência pipeline: {result['pipeline_latency_ms']:.3f} ms")
        print()

    print("COMPARAÇÃO DAS MÉTRICAS")
    print("=" * 104)
    print(
        f"{'Métrica':<16}{'DENSE':>16}{'TF-IDF':>16}"
        f"{'HYBRID RRF':>16}{'FIELD-AWARE':>16}"
    )
    for metric_name in ["Hit@1", "Hit@2", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        print(
            f"{metric_name:<16}"
            f"{metrics['DENSE'][metric_name]:>16.6f}"
            f"{metrics['TF-IDF'][metric_name]:>16.6f}"
            f"{metrics['HYBRID RRF'][metric_name]:>16.6f}"
            f"{metrics['FIELD-AWARE RRF'][metric_name]:>16.6f}"
        )
    print(
        f"{'Mediana rank':<16}"
        f"{metrics['DENSE']['median_rank']:>16.1f}"
        f"{metrics['TF-IDF']['median_rank']:>16.1f}"
        f"{metrics['HYBRID RRF']['median_rank']:>16.1f}"
        f"{metrics['FIELD-AWARE RRF']['median_rank']:>16.1f}"
    )
    print(
        "Nota: na mediana Field-Aware, ausências do Top-10 recebem rank 11; "
        "no MRR elas contribuem 0."
    )

    print()
    print("LATÊNCIA MEDIDA")
    print("=" * 104)
    print("Fits e indexação offline não estão incluídos nos tempos por query.")
    print(
        "Reranking adicional (ms): "
        f"média={np.mean(reranker_latencies):.3f}, "
        f"mediana={median(reranker_latencies):.3f}, "
        f"p95={percentile_95(reranker_latencies):.3f}"
    )
    print(
        "Pipeline Hybrid + Field (ms): "
        f"total={sum(pipeline_latencies):.3f}, "
        f"média={np.mean(pipeline_latencies):.3f}, "
        f"mediana={median(pipeline_latencies):.3f}, "
        f"p95={percentile_95(pipeline_latencies):.3f}"
    )

    improvements = sorted(
        [
        (
            result["hybrid_rank"] - result["field_rank"],
            result,
        )
        for result in results
        if result["field_rank"] is not None
        and result["field_rank"] < result["hybrid_rank"]
        ],
        key=lambda change: change[0],
    )
    regressions = sorted(
        [
        (
            result["field_rank"] - result["hybrid_rank"],
            result,
        )
        for result in results
        if result["field_rank"] is not None
        and result["field_rank"] > result["hybrid_rank"]
        ],
        key=lambda change: change[0],
    )

    print()
    print("MUDANÇAS DE RANK EM RELAÇÃO AO HYBRID RRF")
    print("=" * 104)
    print("Maiores melhorias:")
    if improvements:
        for delta, result in reversed(improvements):
            print(
                f"- {result['query']} | {result['expected_tool']}: "
                f"{result['hybrid_rank']} -> {result['field_rank']} "
                f"(melhora de {delta})"
            )
    else:
        print("- nenhuma")

    print("Pioras:")
    if regressions:
        for delta, result in reversed(regressions):
            print(
                f"- {result['query']} | {result['expected_tool']}: "
                f"{result['hybrid_rank']} -> {result['field_rank']} "
                f"(piora de {delta})"
            )
    else:
        print("- nenhuma")

    print("Fora do Top-10 Hybrid:")
    if outside_candidates:
        for result in outside_candidates:
            print(
                f"- {result['query']} | {result['expected_tool']} | "
                f"rank Hybrid {result['hybrid_rank']}"
            )
    else:
        print("- nenhuma")


if __name__ == "__main__":
    main()
