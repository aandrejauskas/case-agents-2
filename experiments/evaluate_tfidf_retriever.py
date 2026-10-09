"""Compara experimentalmente um Retriever TF-IDF com a baseline dense."""
from statistics import median
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common.data_loader import load_eval_dataset, load_tools
from common.schemas import Tool


HIT_CUTOFFS = [1, 2, 3, 5, 10]
DENSE_BASELINE = {
    "Hit@1": 0.10,
    "Hit@2": 0.20,
    "Hit@3": 0.45,
    "Hit@5": 0.60,
    "Hit@10": 0.90,
    "MRR": 0.313597,
}


def tool_to_text(tool: Tool) -> str:
    """Usa a mesma representação textual do Retriever dense atual."""
    readable_name = tool.name.replace("_", " ")
    return f"{readable_name}. {tool.description} Categoria: {tool.category}."


def find_expected_tool(
    ranking: List[tuple[str, float]], expected_tool: str
) -> tuple[Optional[int], Optional[float]]:
    for rank, (name, score) in enumerate(ranking, start=1):
        if name == expected_tool:
            return rank, score
    return None, None


def calculate_metrics(results: List[dict], tool_count: int) -> Dict[str, object]:
    ranks = [result["rank"] for result in results]
    total = len(ranks)
    hit_metrics = {
        f"Hit@{cutoff}": (
            sum(rank is not None and rank <= cutoff for rank in ranks) / total
            if total
            else 0.0
        )
        for cutoff in HIT_CUTOFFS
    }
    reciprocal_ranks = [0.0 if rank is None else 1 / rank for rank in ranks]
    ranks_for_median = [tool_count + 1 if rank is None else rank for rank in ranks]

    return {
        **hit_metrics,
        "MRR": sum(reciprocal_ranks) / total if total else 0.0,
        "median_rank": median(ranks_for_median) if ranks_for_median else 0.0,
        "distribution": {
            "rank 1": sum(rank == 1 for rank in ranks),
            "rank 2": sum(rank == 2 for rank in ranks),
            "rank 3-5": sum(rank is not None and 3 <= rank <= 5 for rank in ranks),
            "rank 6-10": sum(rank is not None and 6 <= rank <= 10 for rank in ranks),
            "rank >10": sum(rank is None or rank > 10 for rank in ranks),
        },
    }


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_texts = [tool_to_text(tool) for tool in tools]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    tool_matrix = vectorizer.fit_transform(tool_texts)

    results: List[dict] = []
    for item in evaluation_items:
        query_vector = vectorizer.transform([item["query"]])
        scores = cosine_similarity(query_vector, tool_matrix)[0]
        sorted_indices = np.argsort(-scores, kind="stable")
        ranking = [
            (tools[index].name, float(scores[index])) for index in sorted_indices
        ]
        expected_rank, expected_score = find_expected_tool(
            ranking, item["expected_tool"]
        )
        results.append(
            {
                "query": item["query"],
                "expected_tool": item["expected_tool"],
                "rank": expected_rank,
                "score": expected_score,
                "top_5": ranking[:5],
            }
        )

    metrics = calculate_metrics(results, len(tools))
    print_report(
        results,
        metrics,
        vocabulary_size=len(vectorizer.vocabulary_),
        matrix_shape=tool_matrix.shape,
        matrix_nonzero=tool_matrix.nnz,
    )


def print_report(
    results: List[dict],
    metrics: Dict[str, object],
    vocabulary_size: int,
    matrix_shape: tuple[int, int],
    matrix_nonzero: int,
) -> None:
    print("TF-IDF RETRIEVER — DIAGNÓSTICO EXPERIMENTAL")
    print("=" * 76)
    print(f"Tamanho do vocabulário: {vocabulary_size}")
    print(f"Shape da matriz das tools: {matrix_shape}")
    print(f"Elementos não-zero: {matrix_nonzero}")
    print()

    for index, result in enumerate(results, start=1):
        rank_text = "não encontrada" if result["rank"] is None else str(result["rank"])
        score = result["score"]
        score_text = "N/A" if score is None else f"{score:.6f}"
        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   rank: {rank_text}")
        print(f"   score: {score_text}")
        print("   Top-5:")
        for top_rank, (name, top_score) in enumerate(result["top_5"], start=1):
            print(f"      {top_rank}. {name}: {top_score:.6f}")
        print()

    print("MÉTRICAS TF-IDF")
    print("=" * 76)
    print(f"Queries com expected_tool: {len(results)}")
    for cutoff in HIT_CUTOFFS:
        metric_name = f"Hit@{cutoff}"
        value = metrics[metric_name]
        hits = round(value * len(results))
        print(f"{metric_name}: {value:.2%} ({hits}/{len(results)})")
    print(f"MRR: {metrics['MRR']:.6f}")
    print(f"Mediana do rank: {metrics['median_rank']}")

    print()
    print("DISTRIBUIÇÃO DOS RANKS")
    for label, count in metrics["distribution"].items():
        print(f"{label}: {count}")

    print()
    print("COMPARAÇÃO")
    print(f"{'Métrica':<12}{'DENSE ATUAL':>16}{'TF-IDF':>16}")
    for metric_name in ["Hit@1", "Hit@2", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        print(
            f"{metric_name:<12}"
            f"{DENSE_BASELINE[metric_name]:>16.6f}"
            f"{metrics[metric_name]:>16.6f}"
        )


if __name__ == "__main__":
    main()
