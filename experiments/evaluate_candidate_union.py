"""Avalia Candidate Union seguida de um reranker lexical por campos."""
from statistics import median
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_starter.embeddings import TextEmbedder
from common.data_loader import load_eval_dataset, load_tools
from common.schemas import Tool


RRF_C = 60
TOP_K_PER_RETRIEVER = 10
HIT_CUTOFFS = [1, 2, 3, 5, 10]
TFIDF_OPTIONS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "norm": "l2",
}


def tool_to_text(tool: Tool) -> str:
    """Mantém a representação textual usada nos experimentos anteriores."""
    readable_name = tool.name.replace("_", " ")
    return f"{readable_name}. {tool.description} Categoria: {tool.category}."


def rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
    """Converte uma ordenação de índices na posição de cada item."""
    positions = np.empty(len(sorted_indices), dtype=np.int64)
    positions[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
    return positions


def build_candidate_union(
    dense_top: np.ndarray, tfidf_top: np.ndarray
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Une os Top-K sem duplicatas, preservando uma ordem determinística."""
    dense_items = [int(index) for index in dense_top]
    tfidf_items = [int(index) for index in tfidf_top]
    dense_set = set(dense_items)
    tfidf_set = set(tfidf_items)

    # Dense vem primeiro; depois entram apenas os exclusivos do TF-IDF.
    candidates = list(dense_items)
    candidates.extend(index for index in tfidf_items if index not in dense_set)

    return np.asarray(candidates, dtype=np.int64), {
        "both": len(dense_set & tfidf_set),
        "dense_only": len(dense_set - tfidf_set),
        "tfidf_only": len(tfidf_set - dense_set),
    }


def lexical_local_positions(
    query_vector,
    tool_matrix,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """Ranqueia um campo dentro dos candidatos com desempate estável."""
    scores = cosine_similarity(query_vector, tool_matrix[candidate_indices])[0]
    order = np.argsort(-scores, kind="stable")
    return rank_positions(order)


def current_field_rerank(
    query: str,
    hybrid_candidates: np.ndarray,
    name_vectorizer: TfidfVectorizer,
    name_tool_matrix,
    description_vectorizer: TfidfVectorizer,
    description_tool_matrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repete o Field-Aware atual sobre o Top-10 Hybrid RRF."""
    name_query = name_vectorizer.transform([query])
    name_positions = lexical_local_positions(
        name_query, name_tool_matrix, hybrid_candidates
    )
    description_query = description_vectorizer.transform([query])
    description_positions = lexical_local_positions(
        description_query, description_tool_matrix, hybrid_candidates
    )

    base_positions = np.arange(1, len(hybrid_candidates) + 1)
    scores = (
        1 / (RRF_C + base_positions)
        + 1 / (RRF_C + name_positions)
        + 1 / (RRF_C + description_positions)
    )
    order = np.argsort(-scores, kind="stable")
    return order, rank_positions(order)


def union_field_rerank(
    query: str,
    candidate_indices: np.ndarray,
    dense_positions: np.ndarray,
    tfidf_positions: np.ndarray,
    name_vectorizer: TfidfVectorizer,
    name_tool_matrix,
    description_vectorizer: TfidfVectorizer,
    description_tool_matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica o RRF de quatro ranks sobre o conjunto da união."""
    name_query = name_vectorizer.transform([query])
    name_positions = lexical_local_positions(
        name_query, name_tool_matrix, candidate_indices
    )
    description_query = description_vectorizer.transform([query])
    description_positions = lexical_local_positions(
        description_query, description_tool_matrix, candidate_indices
    )

    scores = (
        1 / (RRF_C + dense_positions[candidate_indices])
        + 1 / (RRF_C + tfidf_positions[candidate_indices])
        + 1 / (RRF_C + name_positions)
        + 1 / (RRF_C + description_positions)
    )
    order = np.argsort(-scores, kind="stable")
    return scores, order, rank_positions(order)


def expected_local_rank(
    candidate_indices: np.ndarray,
    local_positions: np.ndarray,
    expected_index: int,
) -> Optional[int]:
    locations = np.flatnonzero(candidate_indices == expected_index)
    if not len(locations):
        return None
    return int(local_positions[int(locations[0])])


def calculate_metrics(
    ranks: List[Optional[int]], missing_ranks: List[int]
) -> Dict[str, float]:
    """Calcula hits, MRR e mediana, penalizando ausências na mediana."""
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
    median_values = [
        missing if rank is None else rank
        for rank, missing in zip(ranks, missing_ranks)
    ]
    metrics["median_rank"] = float(median(median_values)) if median_values else 0.0
    return metrics


def source_label(in_dense: bool, in_tfidf: bool) -> str:
    if in_dense and in_tfidf:
        return "ambos"
    if in_dense:
        return "Dense"
    if in_tfidf:
        return "TF-IDF"
    return "nenhum"


def percentile_95(values: List[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def latency_summary(values: List[float]) -> Dict[str, float]:
    return {
        "total": sum(values),
        "mean": float(np.mean(values)) if values else 0.0,
        "median": float(median(values)) if values else 0.0,
        "p95": percentile_95(values),
    }


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}

    # Todos os fits e índices são construídos uma vez, fora da latência por query.
    tool_texts = [tool_to_text(tool) for tool in tools]
    embedder = TextEmbedder()
    dense_tool_matrix = embedder.embed(tool_texts)

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

    results: List[dict] = []
    for item in evaluation_items:
        query = item["query"]
        expected_tool = item["expected_tool"]
        expected_index = tool_index_by_name[expected_tool]

        retrieval_start = perf_counter()
        dense_query = embedder.embed([query])
        dense_scores = cosine_similarity(dense_query, dense_tool_matrix)[0]
        dense_order = np.argsort(-dense_scores, kind="stable")
        dense_positions = rank_positions(dense_order)

        tfidf_query = combined_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(tfidf_query, combined_tool_matrix)[0]
        tfidf_order = np.argsort(-tfidf_scores, kind="stable")
        tfidf_positions = rank_positions(tfidf_order)
        retrieval_latency_ms = (perf_counter() - retrieval_start) * 1_000

        # Ramo atual: Hybrid RRF Top-10 seguido do Field-Aware existente.
        current_branch_start = perf_counter()
        hybrid_scores = (
            1 / (RRF_C + dense_positions)
            + 1 / (RRF_C + tfidf_positions)
        )
        hybrid_order = np.argsort(-hybrid_scores, kind="stable")
        hybrid_positions = rank_positions(hybrid_order)
        hybrid_candidates = hybrid_order[:TOP_K_PER_RETRIEVER]
        _, current_field_positions = current_field_rerank(
            query,
            hybrid_candidates,
            name_vectorizer,
            name_tool_matrix,
            description_vectorizer,
            description_tool_matrix,
        )
        current_branch_latency_ms = (
            perf_counter() - current_branch_start
        ) * 1_000

        # Novo ramo: união dos Top-10 de cada retriever e RRF de quatro sinais.
        union_start = perf_counter()
        candidate_indices, candidate_counts = build_candidate_union(
            dense_order[:TOP_K_PER_RETRIEVER],
            tfidf_order[:TOP_K_PER_RETRIEVER],
        )
        union_latency_ms = (perf_counter() - union_start) * 1_000

        reranker_start = perf_counter()
        union_scores, union_order, union_positions = union_field_rerank(
            query,
            candidate_indices,
            dense_positions,
            tfidf_positions,
            name_vectorizer,
            name_tool_matrix,
            description_vectorizer,
            description_tool_matrix,
        )
        union_reranker_latency_ms = (
            perf_counter() - reranker_start
        ) * 1_000

        in_dense = bool(
            np.any(dense_order[:TOP_K_PER_RETRIEVER] == expected_index)
        )
        in_tfidf = bool(
            np.any(tfidf_order[:TOP_K_PER_RETRIEVER] == expected_index)
        )
        current_field_rank = expected_local_rank(
            hybrid_candidates, current_field_positions, expected_index
        )
        union_field_rank = expected_local_rank(
            candidate_indices, union_positions, expected_index
        )

        results.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "dense_rank": int(dense_positions[expected_index]),
                "tfidf_rank": int(tfidf_positions[expected_index]),
                "hybrid_rank": int(hybrid_positions[expected_index]),
                "current_field_rank": current_field_rank,
                "union_field_rank": union_field_rank,
                "expected_source": source_label(in_dense, in_tfidf),
                "candidate_count": len(candidate_indices),
                "both_count": candidate_counts["both"],
                "dense_only_count": candidate_counts["dense_only"],
                "tfidf_only_count": candidate_counts["tfidf_only"],
                "union_top_5": [
                    (
                        tool_names[candidate_indices[local_index]],
                        float(union_scores[local_index]),
                    )
                    for local_index in union_order[:5]
                ],
                "union_latency_ms": union_latency_ms,
                "union_reranker_latency_ms": union_reranker_latency_ms,
                "current_pipeline_latency_ms": (
                    retrieval_latency_ms + current_branch_latency_ms
                ),
                "union_pipeline_latency_ms": (
                    retrieval_latency_ms
                    + union_latency_ms
                    + union_reranker_latency_ms
                ),
            }
        )

    full_missing = [len(tools) + 1] * len(results)
    current_missing = [TOP_K_PER_RETRIEVER + 1] * len(results)
    union_missing = [result["candidate_count"] + 1 for result in results]
    metrics = {
        "DENSE": calculate_metrics(
            [result["dense_rank"] for result in results], full_missing
        ),
        "TF-IDF": calculate_metrics(
            [result["tfidf_rank"] for result in results], full_missing
        ),
        "HYBRID RRF": calculate_metrics(
            [result["hybrid_rank"] for result in results], full_missing
        ),
        "FIELD-AWARE": calculate_metrics(
            [result["current_field_rank"] for result in results], current_missing
        ),
        "UNION FIELD": calculate_metrics(
            [result["union_field_rank"] for result in results], union_missing
        ),
    }
    print_report(results, metrics)


def format_rank(rank: Optional[int]) -> str:
    return "fora dos candidatos" if rank is None else str(rank)


def print_latency(label: str, values: List[float]) -> None:
    summary = latency_summary(values)
    print(
        f"{label}: total={summary['total']:.3f} ms, "
        f"média={summary['mean']:.3f} ms, "
        f"mediana={summary['median']:.3f} ms, "
        f"p95={summary['p95']:.3f} ms"
    )


def print_report(results: List[dict], metrics: Dict[str, Dict[str, float]]) -> None:
    total = len(results)
    hybrid_covered = sum(
        result["current_field_rank"] is not None for result in results
    )
    union_covered = sum(
        result["union_field_rank"] is not None for result in results
    )

    print("CANDIDATE UNION + FIELD-AWARE RRF — DIAGNÓSTICO EXPERIMENTAL")
    print("=" * 118)
    print(f"Queries com expected_tool: {total}")
    print(f"Top-K por retriever antes da união: {TOP_K_PER_RETRIEVER}")
    print(f"Constante RRF: C={RRF_C}")
    print(
        "Ordem determinística: Dense Top-10 primeiro; depois exclusivos "
        "do TF-IDF na ordem TF-IDF."
    )
    print("Empates preservam a ordem anterior por ordenação estável.")
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   rank Dense global: {result['dense_rank']}")
        print(f"   rank TF-IDF global: {result['tfidf_rank']}")
        print(f"   rank Hybrid RRF: {result['hybrid_rank']}")
        print(
            "   rank Field-Aware atual: "
            f"{format_rank(result['current_field_rank'])}"
        )
        print(
            "   rank Union Field-Aware: "
            f"{format_rank(result['union_field_rank'])}"
        )
        print(f"   entrada da expected_tool na união: {result['expected_source']}")
        print(
            f"   candidatos: {result['candidate_count']} "
            f"(ambos={result['both_count']}, "
            f"só Dense={result['dense_only_count']}, "
            f"só TF-IDF={result['tfidf_only_count']})"
        )
        print("   Top-5 Union Field-Aware:")
        for rank, (name, score) in enumerate(result["union_top_5"], start=1):
            print(f"      {rank}. {name}: {score:.8f}")
        print()

    print("COMPARAÇÃO DAS MÉTRICAS")
    print("=" * 118)
    print(
        f"{'Métrica':<15}{'DENSE':>14}{'TF-IDF':>14}{'HYBRID':>14}"
        f"{'FIELD ATUAL':>16}{'UNION FIELD':>16}"
    )
    for metric_name in ["Hit@1", "Hit@2", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        print(
            f"{metric_name:<15}"
            f"{metrics['DENSE'][metric_name]:>14.6f}"
            f"{metrics['TF-IDF'][metric_name]:>14.6f}"
            f"{metrics['HYBRID RRF'][metric_name]:>14.6f}"
            f"{metrics['FIELD-AWARE'][metric_name]:>16.6f}"
            f"{metrics['UNION FIELD'][metric_name]:>16.6f}"
        )
    print(
        f"{'Mediana rank':<15}"
        f"{metrics['DENSE']['median_rank']:>14.1f}"
        f"{metrics['TF-IDF']['median_rank']:>14.1f}"
        f"{metrics['HYBRID RRF']['median_rank']:>14.1f}"
        f"{metrics['FIELD-AWARE']['median_rank']:>16.1f}"
        f"{metrics['UNION FIELD']['median_rank']:>16.1f}"
    )
    print()

    print("CANDIDATE COVERAGE")
    print("=" * 118)
    print(
        f"Top-10 Hybrid atual: {hybrid_covered / total:.2%} "
        f"({hybrid_covered}/{total})"
    )
    print(
        f"Candidate Union: {union_covered / total:.2%} "
        f"({union_covered}/{total})"
    )
    candidate_sizes = [result["candidate_count"] for result in results]
    print(f"Tamanho médio da união: {np.mean(candidate_sizes):.2f}")
    print(f"Tamanho mínimo/máximo: {min(candidate_sizes)}/{max(candidate_sizes)}")
    print()

    print("LATÊNCIA MEDIDA")
    print("=" * 118)
    print("Fits e indexações offline não estão incluídos.")
    print_latency(
        "Construção adicional da união",
        [result["union_latency_ms"] for result in results],
    )
    print_latency(
        "Novo reranking",
        [result["union_reranker_latency_ms"] for result in results],
    )
    current_pipeline = [
        result["current_pipeline_latency_ms"] for result in results
    ]
    union_pipeline = [result["union_pipeline_latency_ms"] for result in results]
    print_latency("Pipeline atual Hybrid + Field", current_pipeline)
    print_latency("Pipeline Candidate Union + Field", union_pipeline)
    current_mean = float(np.mean(current_pipeline))
    union_mean = float(np.mean(union_pipeline))
    delta = union_mean - current_mean
    delta_percent = (delta / current_mean * 100) if current_mean else 0.0
    print(
        f"Variação média do pipeline: {delta:+.3f} ms "
        f"({delta_percent:+.2f}%)"
    )
    print()

    recovered = [
        result
        for result in results
        if result["current_field_rank"] is None
        and result["union_field_rank"] is not None
    ]
    lost = [
        result
        for result in results
        if result["current_field_rank"] is not None
        and result["union_field_rank"] is None
    ]
    improved = [
        result
        for result in results
        if result["current_field_rank"] is not None
        and result["union_field_rank"] is not None
        and result["union_field_rank"] < result["current_field_rank"]
    ]
    worsened = [
        result
        for result in results
        if result["current_field_rank"] is not None
        and result["union_field_rank"] is not None
        and result["union_field_rank"] > result["current_field_rank"]
    ]

    print("MUDANÇAS CONTRA O FIELD-AWARE ATUAL")
    print("=" * 118)
    print("Recuperadas pela união:")
    if recovered:
        for result in recovered:
            print(
                f"- {result['query']} | {result['expected_tool']} | "
                f"Union rank {result['union_field_rank']} "
                f"via {result['expected_source']}"
            )
    else:
        print("- nenhuma")

    print("Melhorias de rank:")
    if improved:
        for result in improved:
            print(
                f"- {result['query']} | {result['expected_tool']}: "
                f"{result['current_field_rank']} -> "
                f"{result['union_field_rank']}"
            )
    else:
        print("- nenhuma")

    print("Pioras de rank:")
    if worsened:
        for result in worsened:
            print(
                f"- {result['query']} | {result['expected_tool']}: "
                f"{result['current_field_rank']} -> "
                f"{result['union_field_rank']}"
            )
    else:
        print("- nenhuma")

    print("Perdidas pela união:")
    if lost:
        for result in lost:
            print(f"- {result['query']} | {result['expected_tool']}")
    else:
        print("- nenhuma")


if __name__ == "__main__":
    main()
