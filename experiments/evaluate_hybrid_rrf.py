"""Avalia experimentalmente a fusão dos rankings dense e TF-IDF com RRF."""
from statistics import median
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_starter.embeddings import TextEmbedder
from common.data_loader import load_eval_dataset, load_tools
from common.schemas import Tool


RRF_C = 60
HIT_CUTOFFS = [1, 2, 3, 5, 10]


def tool_to_text(tool: Tool) -> str:
    """Usa a mesma representação textual dos experimentos anteriores."""
    readable_name = tool.name.replace("_", " ")
    return f"{readable_name}. {tool.description} Categoria: {tool.category}."


def rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
    """Converte uma ordenação de índices na posição de cada tool no ranking."""
    positions = np.empty(len(sorted_indices), dtype=np.int64)
    positions[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
    return positions


def calculate_metrics(ranks: List[int]) -> Dict[str, float]:
    total = len(ranks)
    metrics = {
        f"Hit@{cutoff}": sum(rank <= cutoff for rank in ranks) / total
        for cutoff in HIT_CUTOFFS
    }
    metrics["MRR"] = sum(1 / rank for rank in ranks) / total
    metrics["median_rank"] = float(median(ranks))
    return metrics


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}
    tool_texts = [tool_to_text(tool) for tool in tools]

    embedder = TextEmbedder()
    dense_tool_matrix = embedder.embed(tool_texts)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    sparse_tool_matrix = vectorizer.fit_transform(tool_texts)

    results: List[dict] = []
    for item in evaluation_items:
        query = item["query"]
        expected_tool = item["expected_tool"]
        expected_index = tool_index_by_name[expected_tool]

        dense_query = embedder.embed([query])
        dense_scores = cosine_similarity(dense_query, dense_tool_matrix)[0]
        dense_order = np.argsort(-dense_scores, kind="stable")
        dense_positions = rank_positions(dense_order)

        sparse_query = vectorizer.transform([query])
        sparse_scores = cosine_similarity(sparse_query, sparse_tool_matrix)[0]
        sparse_order = np.argsort(-sparse_scores, kind="stable")
        sparse_positions = rank_positions(sparse_order)

        # RRF combina posições, evitando somar scores de escalas incompatíveis.
        rrf_scores = (
            1 / (RRF_C + dense_positions)
            + 1 / (RRF_C + sparse_positions)
        )
        rrf_order = np.argsort(-rrf_scores, kind="stable")
        rrf_positions = rank_positions(rrf_order)

        results.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "dense_rank": int(dense_positions[expected_index]),
                "tfidf_rank": int(sparse_positions[expected_index]),
                "rrf_rank": int(rrf_positions[expected_index]),
                "rrf_top_5": [
                    (tool_names[index], float(rrf_scores[index]))
                    for index in rrf_order[:5]
                ],
            }
        )

    metrics = {
        "DENSE": calculate_metrics([result["dense_rank"] for result in results]),
        "TF-IDF": calculate_metrics([result["tfidf_rank"] for result in results]),
        "HYBRID RRF": calculate_metrics([result["rrf_rank"] for result in results]),
    }
    print_report(results, metrics)


def print_report(results: List[dict], metrics: Dict[str, Dict[str, float]]) -> None:
    print("HYBRID RETRIEVAL — RECIPROCAL RANK FUSION")
    print("=" * 84)
    print(f"Queries com expected_tool: {len(results)}")
    print(f"Constante RRF C: {RRF_C}")
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   rank dense: {result['dense_rank']}")
        print(f"   rank TF-IDF: {result['tfidf_rank']}")
        print(f"   rank RRF: {result['rrf_rank']}")
        print("   Top-5 RRF:")
        for rank, (name, score) in enumerate(result["rrf_top_5"], start=1):
            print(f"      {rank}. {name}: {score:.8f}")
        print()

    print("COMPARAÇÃO DAS MÉTRICAS")
    print("=" * 84)
    print(f"{'Métrica':<16}{'DENSE':>16}{'TF-IDF':>16}{'HYBRID RRF':>16}")
    for metric_name in ["Hit@1", "Hit@2", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        print(
            f"{metric_name:<16}"
            f"{metrics['DENSE'][metric_name]:>16.6f}"
            f"{metrics['TF-IDF'][metric_name]:>16.6f}"
            f"{metrics['HYBRID RRF'][metric_name]:>16.6f}"
        )
    print(
        f"{'Mediana rank':<16}"
        f"{metrics['DENSE']['median_rank']:>16.1f}"
        f"{metrics['TF-IDF']['median_rank']:>16.1f}"
        f"{metrics['HYBRID RRF']['median_rank']:>16.1f}"
    )


if __name__ == "__main__":
    main()
