"""Avalia um Cross-Encoder sobre a Candidate Union dos retrievers atuais."""
from statistics import median
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_starter.embeddings import TextEmbedder
from experiments.evaluate_candidate_union import (
    RRF_C,
    TFIDF_OPTIONS,
    TOP_K_PER_RETRIEVER,
    build_candidate_union,
    calculate_metrics,
    current_field_rerank,
    expected_local_rank,
    latency_summary,
    rank_positions,
    source_label,
    tool_to_text,
    union_field_rerank,
)
from common.data_loader import load_eval_dataset, load_tools


# Cross-Encoder multilíngue baseado em MiniLMv2 e treinado para relevância.
CROSS_ENCODER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
HIT_CUTOFFS = [1, 2, 3, 5, 10]


def percentile_95(values: List[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def format_rank(rank: Optional[int]) -> str:
    return "fora dos candidatos" if rank is None else str(rank)


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}
    tool_texts = [tool_to_text(tool) for tool in tools]

    # Fits e construção dos índices ficam fora de qualquer latência por query.
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

    initialization_start = perf_counter()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_ID)
    # O warm-up absorve o cold start junto do carregamento, não nas queries.
    cross_encoder.predict(
        [("consulta de teste", "ferramenta de teste")],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    initialization_latency_ms = (perf_counter() - initialization_start) * 1_000
    device = str(next(cross_encoder.model.parameters()).device)

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

        union_start = perf_counter()
        candidate_indices, _ = build_candidate_union(
            dense_order[:TOP_K_PER_RETRIEVER],
            tfidf_order[:TOP_K_PER_RETRIEVER],
        )
        union_latency_ms = (perf_counter() - union_start) * 1_000

        cross_encoder_pairs = [
            (query, tool_texts[tool_index])
            for tool_index in candidate_indices
        ]
        cross_encoder_start = perf_counter()
        cross_encoder_scores = np.asarray(
            cross_encoder.predict(
                cross_encoder_pairs,
                batch_size=len(cross_encoder_pairs),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)
        cross_encoder_latency_ms = (
            perf_counter() - cross_encoder_start
        ) * 1_000
        cross_encoder_order = np.argsort(
            -cross_encoder_scores, kind="stable"
        )
        cross_encoder_positions = rank_positions(cross_encoder_order)

        # Baselines calculadas fora da composição de latência do novo pipeline.
        hybrid_scores = (
            1 / (RRF_C + dense_positions)
            + 1 / (RRF_C + tfidf_positions)
        )
        hybrid_order = np.argsort(-hybrid_scores, kind="stable")
        hybrid_candidates = hybrid_order[:TOP_K_PER_RETRIEVER]
        _, current_field_positions = current_field_rerank(
            query,
            hybrid_candidates,
            name_vectorizer,
            name_tool_matrix,
            description_vectorizer,
            description_tool_matrix,
        )
        _, _, union_field_positions = union_field_rerank(
            query,
            candidate_indices,
            dense_positions,
            tfidf_positions,
            name_vectorizer,
            name_tool_matrix,
            description_vectorizer,
            description_tool_matrix,
        )

        in_dense = bool(
            np.any(dense_order[:TOP_K_PER_RETRIEVER] == expected_index)
        )
        in_tfidf = bool(
            np.any(tfidf_order[:TOP_K_PER_RETRIEVER] == expected_index)
        )
        results.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "dense_rank": int(dense_positions[expected_index]),
                "current_field_rank": expected_local_rank(
                    hybrid_candidates,
                    current_field_positions,
                    expected_index,
                ),
                "union_field_rank": expected_local_rank(
                    candidate_indices,
                    union_field_positions,
                    expected_index,
                ),
                "cross_encoder_rank": expected_local_rank(
                    candidate_indices,
                    cross_encoder_positions,
                    expected_index,
                ),
                "expected_source": source_label(in_dense, in_tfidf),
                "candidate_count": len(candidate_indices),
                "cross_encoder_top_5": [
                    (
                        tool_names[candidate_indices[local_index]],
                        float(cross_encoder_scores[local_index]),
                    )
                    for local_index in cross_encoder_order[:5]
                ],
                "retrieval_latency_ms": retrieval_latency_ms,
                "union_latency_ms": union_latency_ms,
                "cross_encoder_latency_ms": cross_encoder_latency_ms,
                "pipeline_latency_ms": (
                    retrieval_latency_ms
                    + union_latency_ms
                    + cross_encoder_latency_ms
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
        "FIELD-AWARE": calculate_metrics(
            [result["current_field_rank"] for result in results], current_missing
        ),
        "UNION FIELD": calculate_metrics(
            [result["union_field_rank"] for result in results], union_missing
        ),
        "CROSS-ENCODER": calculate_metrics(
            [result["cross_encoder_rank"] for result in results], union_missing
        ),
    }
    print_report(
        results,
        metrics,
        initialization_latency_ms=initialization_latency_ms,
        device=device,
    )


def print_latency(label: str, values: List[float]) -> None:
    summary = latency_summary(values)
    print(
        f"{label}: média={summary['mean']:.3f} ms, "
        f"mediana={summary['median']:.3f} ms, "
        f"p95={summary['p95']:.3f} ms"
    )


def print_report(
    results: List[dict],
    metrics: Dict[str, Dict[str, float]],
    initialization_latency_ms: float,
    device: str,
) -> None:
    total = len(results)
    covered = sum(
        result["cross_encoder_rank"] is not None for result in results
    )

    print("CANDIDATE UNION + CROSS-ENCODER — DIAGNÓSTICO EXPERIMENTAL")
    print("=" * 112)
    print(f"Modelo: {CROSS_ENCODER_MODEL_ID}")
    print(f"Dispositivo: {device}")
    print(
        "Inicialização + warm-up fora da latência por query: "
        f"{initialization_latency_ms:.3f} ms"
    )
    print(f"Queries com expected_tool: {total}")
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   rank Dense: {result['dense_rank']}")
        print(
            "   rank Field-Aware atual: "
            f"{format_rank(result['current_field_rank'])}"
        )
        print(
            "   rank Candidate Union + Field: "
            f"{format_rank(result['union_field_rank'])}"
        )
        print(
            "   rank Candidate Union + Cross-Encoder: "
            f"{format_rank(result['cross_encoder_rank'])}"
        )
        print(f"   entrada na união: {result['expected_source']}")
        print(f"   pares avaliados no batch: {result['candidate_count']}")
        print("   Top-5 Cross-Encoder:")
        for rank, (name, score) in enumerate(
            result["cross_encoder_top_5"], start=1
        ):
            print(f"      {rank}. {name}: {score:.8f}")
        print()

    print("COMPARAÇÃO DAS MÉTRICAS")
    print("=" * 112)
    print(
        f"{'Métrica':<15}{'DENSE':>16}{'FIELD ATUAL':>18}"
        f"{'UNION FIELD':>18}{'CROSS-ENCODER':>20}"
    )
    for metric_name in ["Hit@1", "Hit@2", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        print(
            f"{metric_name:<15}"
            f"{metrics['DENSE'][metric_name]:>16.6f}"
            f"{metrics['FIELD-AWARE'][metric_name]:>18.6f}"
            f"{metrics['UNION FIELD'][metric_name]:>18.6f}"
            f"{metrics['CROSS-ENCODER'][metric_name]:>20.6f}"
        )
    print(
        f"{'Mediana rank':<15}"
        f"{metrics['DENSE']['median_rank']:>16.1f}"
        f"{metrics['FIELD-AWARE']['median_rank']:>18.1f}"
        f"{metrics['UNION FIELD']['median_rank']:>18.1f}"
        f"{metrics['CROSS-ENCODER']['median_rank']:>20.1f}"
    )
    print()

    print("CANDIDATE COVERAGE E CUSTO LOCAL")
    print("=" * 112)
    print(f"Candidate coverage: {covered / total:.2%} ({covered}/{total})")
    print(
        "Número médio de pares por query: "
        f"{np.mean([result['candidate_count'] for result in results]):.2f}"
    )
    print(f"Dispositivo utilizado: {device}")
    print("Custo monetário de API: não aplicável; inferência local.")
    print()

    print("LATÊNCIA POR QUERY")
    print("=" * 112)
    print("Download, carregamento, warm-up, fits e indexações não estão incluídos.")
    print_latency(
        "Embedding/query retrieval",
        [result["retrieval_latency_ms"] for result in results],
    )
    print_latency(
        "Candidate Union",
        [result["union_latency_ms"] for result in results],
    )
    print_latency(
        "Inferência Cross-Encoder",
        [result["cross_encoder_latency_ms"] for result in results],
    )
    print_latency(
        "Pipeline total",
        [result["pipeline_latency_ms"] for result in results],
    )
    print()

    new_top_2 = [
        result
        for result in results
        if (result["current_field_rank"] is None or result["current_field_rank"] > 2)
        and result["cross_encoder_rank"] is not None
        and result["cross_encoder_rank"] <= 2
    ]
    lost_top_2 = [
        result
        for result in results
        if result["current_field_rank"] is not None
        and result["current_field_rank"] <= 2
        and (
            result["cross_encoder_rank"] is None
            or result["cross_encoder_rank"] > 2
        )
    ]
    improved = [
        result
        for result in results
        if result["current_field_rank"] is not None
        and result["cross_encoder_rank"] is not None
        and result["cross_encoder_rank"] < result["current_field_rank"]
    ]
    worsened = [
        result
        for result in results
        if result["current_field_rank"] is not None
        and result["cross_encoder_rank"] is not None
        and result["cross_encoder_rank"] > result["current_field_rank"]
    ]

    print("MUDANÇAS CONTRA O FIELD-AWARE ATUAL")
    print("=" * 112)
    print(f"Novas queries no Top-2: {len(new_top_2)}")
    for result in new_top_2:
        print(
            f"- {result['query']} | {result['expected_tool']}: "
            f"{format_rank(result['current_field_rank'])} -> "
            f"{result['cross_encoder_rank']}"
        )
    print(f"Queries que saíram do Top-2: {len(lost_top_2)}")
    for result in lost_top_2:
        print(
            f"- {result['query']} | {result['expected_tool']}: "
            f"{result['current_field_rank']} -> "
            f"{format_rank(result['cross_encoder_rank'])}"
        )

    print("Melhorias de rank:")
    if improved:
        for result in improved:
            print(
                f"- {result['query']} | {result['expected_tool']}: "
                f"{result['current_field_rank']} -> "
                f"{result['cross_encoder_rank']}"
            )
    else:
        print("- nenhuma")

    print("Pioras de rank:")
    if worsened:
        for result in worsened:
            print(
                f"- {result['query']} | {result['expected_tool']}: "
                f"{result['current_field_rank']} -> "
                f"{result['cross_encoder_rank']}"
            )
    else:
        print("- nenhuma")


if __name__ == "__main__":
    main()
