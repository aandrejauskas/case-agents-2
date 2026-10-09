"""Avalia a posição da tool esperada no ranking completo do Retriever."""
from typing import Dict, List, Optional

from candidate_starter.retrieval import ToolRetriever
from common.data_loader import load_eval_dataset, load_tools


HIT_CUTOFFS = [1, 2, 3, 5, 10]


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    retriever = ToolRetriever().fit(tools)

    results: List[dict] = []
    for item in evaluation_items:
        retrieval = retriever.search(item["query"], k=len(tools))
        expected_rank, expected_score = find_expected_tool(
            retrieval.matches,
            item["expected_tool"],
        )
        results.append(
            {
                "query": item["query"],
                "expected_tool": item["expected_tool"],
                "rank": expected_rank,
                "score": expected_score,
                "top_5": retrieval.matches[:5],
            }
        )

    print_results(results)


def find_expected_tool(matches, expected_tool: str) -> tuple[Optional[int], Optional[float]]:
    for rank, match in enumerate(matches, start=1):
        if match.name == expected_tool:
            return rank, match.score
    return None, None


def print_results(results: List[dict]) -> None:
    for index, result in enumerate(results, start=1):
        rank_text = "não encontrada" if result["rank"] is None else str(result["rank"])
        score = result["score"]
        score_text = "N/A" if score is None else f"{score:.6f}"

        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   rank: {rank_text}")
        print(f"   score: {score_text}")
        print("   Top-5:")
        for top_rank, match in enumerate(result["top_5"], start=1):
            print(f"      {top_rank}. {match.name}: {match.score:.6f}")
        print()

    total = len(results)
    ranks = [result["rank"] for result in results]
    reciprocal_ranks = [0.0 if rank is None else 1 / rank for rank in ranks]

    print("MÉTRICAS DO RANKING COMPLETO")
    print("=" * 72)
    print(f"Queries com expected_tool: {total}")
    for cutoff in HIT_CUTOFFS:
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        metric_name = "Top-1 Accuracy / Hit@1" if cutoff == 1 else f"Hit@{cutoff}"
        value = hits / total if total else 0.0
        print(f"{metric_name}: {value:.2%} ({hits}/{total})")

    mrr = sum(reciprocal_ranks) / total if total else 0.0
    print(f"MRR: {mrr:.6f}")

    distribution: Dict[str, int] = {
        "rank 1": sum(rank == 1 for rank in ranks),
        "rank 2": sum(rank == 2 for rank in ranks),
        "rank 3-5": sum(rank is not None and 3 <= rank <= 5 for rank in ranks),
        "rank 6-10": sum(rank is not None and 6 <= rank <= 10 for rank in ranks),
        "rank >10": sum(rank is None or rank > 10 for rank in ranks),
    }

    print()
    print("DISTRIBUIÇÃO DOS RANKS")
    for label, count in distribution.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
