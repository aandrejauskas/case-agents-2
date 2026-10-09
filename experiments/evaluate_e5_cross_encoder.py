"""Reranqueia exclusivamente o Top-5 do E5 Small com um Cross-Encoder.

Experimento diagnóstico isolado. Nenhum candidato fora do Top-5 dense pode
entrar no ranking final e nenhum componente de produção importa este módulo.
"""
from statistics import median
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_starter.retrieval import ToolRetriever
from common.data_loader import load_eval_dataset, load_tools


E5_MODEL_ID = "intfloat/multilingual-e5-small"
CROSS_ENCODER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CANDIDATE_COUNT = 5
HIT_CUTOFFS = [1, 2, 3, 5]


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1_000


def latency_summary(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def calculate_metrics(ranks: List[Optional[int]]) -> Dict[str, float]:
    total = len(ranks)
    metrics = {
        f"Hit@{cutoff}": sum(
            rank is not None and rank <= cutoff for rank in ranks
        )
        / total
        for cutoff in HIT_CUTOFFS
    }
    metrics["MRR"] = sum(
        0.0 if rank is None else 1.0 / rank for rank in ranks
    ) / total
    return metrics


def format_rank(rank: Optional[int]) -> str:
    return "fora do Top-5" if rank is None else str(rank)


def print_latency(label: str, values: List[float]) -> None:
    summary = latency_summary(values)
    print(
        f"{label:<31} média={summary['mean']:.3f} ms | "
        f"mediana={summary['median']:.3f} ms | p95={summary['p95']:.3f} ms"
    )


def main() -> None:
    tools = load_tools()
    items = [item for item in load_eval_dataset() if item.get("expected_tool")]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}
    # Usa literalmente a representação da implementação final.
    tool_texts = [ToolRetriever._tool_to_text(tool) for tool in tools]

    e5_initialization_start = perf_counter()
    e5 = SentenceTransformer(E5_MODEL_ID, device="cpu")
    e5_initialization_ms = elapsed_ms(e5_initialization_start)

    tool_embedding_start = perf_counter()
    tool_embeddings = e5.encode(
        [f"passage: {text}" for text in tool_texts],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    tool_embedding_ms = elapsed_ms(tool_embedding_start)

    cross_initialization_start = perf_counter()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_ID, device="cpu")
    # O warm-up fica junto da inicialização, fora da latência por query.
    cross_encoder.predict(
        [("consulta de aquecimento", "ferramenta de aquecimento")],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    cross_initialization_ms = elapsed_ms(cross_initialization_start)
    cross_device = str(next(cross_encoder.model.parameters()).device)

    results = []
    for item in items:
        query = item["query"]
        expected_tool = item["expected_tool"]
        expected_index = tool_index_by_name[expected_tool]

        query_embedding_start = perf_counter()
        query_embedding = e5.encode(
            [f"query: {query}"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding_ms = elapsed_ms(query_embedding_start)

        cosine_start = perf_counter()
        e5_scores = cosine_similarity(query_embedding, tool_embeddings)[0]
        e5_order = np.argsort(-e5_scores, kind="stable")
        candidates = e5_order[:CANDIDATE_COUNT]
        e5_rank = int(np.flatnonzero(e5_order == expected_index)[0]) + 1
        cosine_ms = elapsed_ms(cosine_start)

        cross_start = perf_counter()
        pairs = [(query, tool_texts[int(index)]) for index in candidates]
        cross_scores = np.asarray(
            cross_encoder.predict(
                pairs,
                batch_size=CANDIDATE_COUNT,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ).reshape(-1)
        local_order = np.argsort(-cross_scores, kind="stable")
        reranked_candidates = candidates[local_order]
        cross_ms = elapsed_ms(cross_start)

        expected_position = np.flatnonzero(reranked_candidates == expected_index)
        cross_rank = (
            int(expected_position[0]) + 1 if expected_position.size else None
        )

        results.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "e5_rank": e5_rank,
                "cross_rank": cross_rank,
                "e5_top_5": [tool_names[int(index)] for index in candidates],
                "cross_top_5": [
                    (
                        tool_names[int(candidates[local_index])],
                        float(cross_scores[local_index]),
                    )
                    for local_index in local_order
                ],
                "query_embedding_ms": query_embedding_ms,
                "cosine_ms": cosine_ms,
                "cross_ms": cross_ms,
                "pipeline_ms": query_embedding_ms + cosine_ms + cross_ms,
            }
        )

    print_report(
        results,
        e5_initialization_ms=e5_initialization_ms,
        tool_embedding_ms=tool_embedding_ms,
        cross_initialization_ms=cross_initialization_ms,
        cross_device=cross_device,
        tool_count=len(tools),
    )


def print_report(
    results: List[dict],
    e5_initialization_ms: float,
    tool_embedding_ms: float,
    cross_initialization_ms: float,
    cross_device: str,
    tool_count: int,
) -> None:
    total = len(results)
    e5_metrics = calculate_metrics([result["e5_rank"] for result in results])
    cross_metrics = calculate_metrics(
        [result["cross_rank"] for result in results]
    )

    print("E5 SMALL TOP-5 + CROSS-ENCODER — DIAGNÓSTICO DE FINE RANKING")
    print("=" * 116)
    print(f"E5: {E5_MODEL_ID}")
    print(f"Cross-Encoder: {CROSS_ENCODER_MODEL_ID}")
    print(f"Dispositivo do Cross-Encoder: {cross_device}")
    print(f"Catálogo: {tool_count} tools | Queries: {total}")
    print(
        "AVISO: o eval já foi observado repetidamente. Este resultado é apenas "
        "diagnóstico; cada query representa 5 pontos percentuais."
    )
    print()

    print("MÉTRICAS")
    print("=" * 116)
    print(f"{'Métrica':<14}{'E5_SMALL_DENSE':>24}{'E5_TOP5_CROSS_ENCODER':>30}")
    for metric_name in ["Hit@1", "Hit@2", "Hit@3", "Hit@5"]:
        e5_hits = round(e5_metrics[metric_name] * total)
        cross_hits = round(cross_metrics[metric_name] * total)
        print(
            f"{metric_name:<14}"
            f"{e5_metrics[metric_name]:>17.2%} ({e5_hits:>2}/{total})"
            f"{cross_metrics[metric_name]:>23.2%} ({cross_hits:>2}/{total})"
        )
    print(
        f"{'MRR':<14}{e5_metrics['MRR']:>24.6f}"
        f"{cross_metrics['MRR']:>30.6f}"
    )
    print()

    print("RANKING ANTES E DEPOIS POR QUERY")
    print("=" * 116)
    for index, result in enumerate(results, start=1):
        print(f"{index}. Query: {result['query']}")
        print("   E5 Top-5:")
        for rank, name in enumerate(result["e5_top_5"], start=1):
            print(f"      {rank}. {name}")
        print("   Cross-Encoder reranked:")
        for rank, (name, score) in enumerate(result["cross_top_5"], start=1):
            print(f"      {rank}. {name} - {score:.8f}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   rank antes: {result['e5_rank']}")
        print(f"   rank depois: {format_rank(result['cross_rank'])}")
        print()

    misses = [result for result in results if result["e5_rank"] > 2]
    recovered = [
        result
        for result in misses
        if result["cross_rank"] is not None and result["cross_rank"] <= 2
    ]
    regressions = [
        result
        for result in results
        if result["e5_rank"] <= 2
        and (result["cross_rank"] is None or result["cross_rank"] > 2)
    ]

    print("ANÁLISE DOS MISSES DO TOP-2 E5")
    print("=" * 116)
    for result in misses:
        status = (
            "RECUPERADA"
            if result["cross_rank"] is not None and result["cross_rank"] <= 2
            else "NÃO ALTERADA"
        )
        print(
            f"{status}: {result['query']} | {result['expected_tool']} | "
            f"E5={result['e5_rank']} | Cross={format_rank(result['cross_rank'])}"
        )
    print()

    print("REGRESSÕES DO TOP-2")
    print("=" * 116)
    if regressions:
        for result in regressions:
            print(
                f"REGRESSÃO: {result['query']} | {result['expected_tool']} | "
                f"E5={result['e5_rank']} | Cross={format_rank(result['cross_rank'])}"
            )
    else:
        print("Nenhuma regressão.")
    print()

    e5_top_2_hits = round(e5_metrics["Hit@2"] * total)
    cross_top_2_hits = round(cross_metrics["Hit@2"] * total)
    print("SALDO TOP-2")
    print("=" * 116)
    print(f"Hits E5 Top-2: {e5_top_2_hits}/{total}")
    print(f"Hits Cross-Encoder Top-2: {cross_top_2_hits}/{total}")
    print(f"Recuperadas: {len(recovered)}")
    print(f"Regressões: {len(regressions)}")
    print(f"Ganho líquido: {cross_top_2_hits - e5_top_2_hits:+d} query(s)")
    print()

    print("LATÊNCIA")
    print("=" * 116)
    print(f"Inicialização E5: {e5_initialization_ms:.3f} ms")
    print(f"Embeddings das tools: {tool_embedding_ms:.3f} ms")
    print(
        "Inicialização + warm-up Cross-Encoder: "
        f"{cross_initialization_ms:.3f} ms"
    )
    print_latency(
        "E5 query embedding",
        [result["query_embedding_ms"] for result in results],
    )
    print_latency(
        "Busca cosine + Top-5",
        [result["cosine_ms"] for result in results],
    )
    print_latency(
        "Cross-Encoder Top-5",
        [result["cross_ms"] for result in results],
    )
    print_latency(
        "Pipeline E5 + reranker",
        [result["pipeline_ms"] for result in results],
    )


if __name__ == "__main__":
    main()
