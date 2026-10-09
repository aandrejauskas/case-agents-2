"""Avalia MMR no Top-5 do E5 Small, preservando obrigatoriamente o Top-1."""
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common.data_loader import load_eval_dataset, load_tools
from common.schemas import Tool


MODEL_ID = "intfloat/multilingual-e5-small"
MMR_LAMBDA = 0.70
CANDIDATE_POOL_SIZE = 5


def tool_to_text(tool: Tool) -> str:
    """Usa a mesma representação textual dos experimentos E5 anteriores."""
    readable_name = tool.name.replace("_", " ")
    return f"{readable_name}. {tool.description} Categoria: {tool.category}."


def rank_positions(sorted_indices: np.ndarray) -> np.ndarray:
    """Converte a ordenação global na posição, iniciando em 1, de cada tool."""
    positions = np.empty(len(sorted_indices), dtype=np.int64)
    positions[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
    return positions


def choose_second_with_mmr(
    dense_top_5: np.ndarray,
    dense_scores: np.ndarray,
    tool_embeddings: np.ndarray,
) -> tuple[int, Dict[int, dict]]:
    """Escolhe o segundo item entre as posições 2-5; o Top-1 é imutável."""
    first_index = int(dense_top_5[0])
    details: Dict[int, dict] = {}

    # Os embeddings estão normalizados, mas cosine_similarity explicita a
    # definição da redundância e mantém o experimento fácil de auditar.
    for candidate_index_value in dense_top_5[1:]:
        candidate_index = int(candidate_index_value)
        relevance = float(dense_scores[candidate_index])
        redundancy = float(
            cosine_similarity(
                tool_embeddings[candidate_index].reshape(1, -1),
                tool_embeddings[first_index].reshape(1, -1),
            )[0, 0]
        )
        weighted_relevance = MMR_LAMBDA * relevance
        redundancy_penalty = (1 - MMR_LAMBDA) * redundancy
        details[candidate_index] = {
            "relevance": relevance,
            "weighted_relevance": weighted_relevance,
            "redundancy": redundancy,
            "redundancy_penalty": redundancy_penalty,
            "mmr_score": weighted_relevance - redundancy_penalty,
        }

    # max preserva o primeiro candidato em caso de empate. Como details segue
    # a ordem Dense original, o desempate é estável e determinístico.
    second_index = max(
        details,
        key=lambda candidate_index: details[candidate_index]["mmr_score"],
    )
    return second_index, details


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}

    model = SentenceTransformer(MODEL_ID, device="cpu")
    tool_embeddings = model.encode(
        [f"passage: {tool_to_text(tool)}" for tool in tools],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results: List[dict] = []
    for item in evaluation_items:
        query = item["query"]
        expected_tool = item["expected_tool"]
        expected_index = tool_index_by_name[expected_tool]

        query_embedding = model.encode(
            [f"query: {query}"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dense_scores = cosine_similarity(query_embedding, tool_embeddings)[0]
        dense_order = np.argsort(-dense_scores, kind="stable")
        dense_positions = rank_positions(dense_order)
        dense_top_5 = dense_order[:CANDIDATE_POOL_SIZE]

        second_index, candidate_details = choose_second_with_mmr(
            dense_top_5,
            dense_scores,
            tool_embeddings,
        )
        first_index = int(dense_top_5[0])
        original_top_2 = [int(index) for index in dense_top_5[:2]]
        mmr_top_2 = [first_index, second_index]
        expected_rank = int(dense_positions[expected_index])
        dense_hit_2 = expected_index in original_top_2
        mmr_hit_2 = expected_index in mmr_top_2

        results.append(
            {
                "query": query,
                "expected_tool": expected_tool,
                "expected_index": expected_index,
                "expected_rank": expected_rank,
                "original_top_2": [tool_names[index] for index in original_top_2],
                "mmr_top_2": [tool_names[index] for index in mmr_top_2],
                "dense_hit_1": expected_rank == 1,
                "dense_hit_2": dense_hit_2,
                "mmr_hit_1": expected_index == first_index,
                "mmr_hit_2": mmr_hit_2,
                "promoted": not dense_hit_2 and mmr_hit_2,
                "lost": dense_hit_2 and not mmr_hit_2,
                "expected_mmr": candidate_details.get(expected_index),
                "selected_second_tool": tool_names[second_index],
                "selected_second_mmr": candidate_details[second_index],
            }
        )

    print_report(results)


def format_top_2(names: List[str]) -> str:
    return f"[{names[0]}, {names[1]}]"


def print_report(results: List[dict]) -> None:
    total = len(results)
    dense_hit_1 = sum(result["dense_hit_1"] for result in results)
    dense_hit_2 = sum(result["dense_hit_2"] for result in results)
    mmr_hit_1 = sum(result["mmr_hit_1"] for result in results)
    mmr_hit_2 = sum(result["mmr_hit_2"] for result in results)
    near_misses = [
        result for result in results if 3 <= result["expected_rank"] <= 5
    ]
    recovered_near_misses = [
        result for result in near_misses if result["promoted"]
    ]
    lost_existing_hits = [result for result in results if result["lost"]]

    print("E5 SMALL DENSE TOP-5 -> MMR -> TOP-2")
    print("=" * 112)
    print(f"Modelo: {MODEL_ID}")
    print("Query prefix: query:")
    print("Tool prefix: passage:")
    print("Embeddings normalizados: sim")
    print(f"Candidate pool: Top-{CANDIDATE_POOL_SIZE} Dense")
    print(f"Lambda fixo: {MMR_LAMBDA:.2f}")
    print("Top-1 permanece obrigatoriamente igual ao Dense.")
    print(
        "O segundo resultado é escolhido somente entre as posições 2-5 "
        "do ranking Dense."
    )
    print()

    print("MÉTRICAS GERAIS")
    print("=" * 112)
    print(f"Queries avaliadas: {total}")
    print(
        f"Dense Hit@1: {dense_hit_1 / total:.2%} "
        f"({dense_hit_1}/{total})"
    )
    print(
        f"MMR Hit@1:   {mmr_hit_1 / total:.2%} "
        f"({mmr_hit_1}/{total})"
    )
    print(
        f"Dense Hit@2: {dense_hit_2 / total:.2%} "
        f"({dense_hit_2}/{total})"
    )
    print(
        f"MMR Hit@2:   {mmr_hit_2 / total:.2%} "
        f"({mmr_hit_2}/{total})"
    )
    print(f"Variação líquida de Hit@2: {mmr_hit_2 - dense_hit_2:+d} query(s)")
    print()

    print("ANÁLISE DOS NEAR MISSES (RANKS 3-5)")
    print("=" * 112)
    print(f"Near misses encontrados: {len(near_misses)}")
    for index, result in enumerate(near_misses, start=1):
        expected_mmr = result["expected_mmr"]
        status = "PROMOVIDA" if result["promoted"] else "CONTINUOU FORA"
        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   expected rank original: {result['expected_rank']}")
        print(f"   Top-2 original: {format_top_2(result['original_top_2'])}")
        print(f"   Top-2 MMR: {format_top_2(result['mmr_top_2'])}")
        print(f"   resultado: {status}")
        print(f"   relevância cosine: {expected_mmr['relevance']:.6f}")
        print(
            f"   relevância ponderada (0.70 × relevância): "
            f"{expected_mmr['weighted_relevance']:.6f}"
        )
        print(f"   redundância cosine com Top-1: {expected_mmr['redundancy']:.6f}")
        print(
            f"   penalidade de redundância (0.30 × redundância): "
            f"{expected_mmr['redundancy_penalty']:.6f}"
        )
        print(f"   score MMR final: {expected_mmr['mmr_score']:.6f}")
        print(
            f"   segunda selecionada: {result['selected_second_tool']} "
            f"(MMR={result['selected_second_mmr']['mmr_score']:.6f})"
        )
        print()

    near_miss_total = len(near_misses)
    recovery_rate = (
        len(recovered_near_misses) / near_miss_total
        if near_miss_total
        else 0.0
    )
    print("RESUMO DE RECUPERAÇÃO")
    print("=" * 112)
    print(
        "Near-miss recovery rate: "
        f"{recovery_rate:.2%} "
        f"({len(recovered_near_misses)}/{near_miss_total})"
    )
    print(f"Hits@2 existentes perdidos pelo MMR: {len(lost_existing_hits)}")
    if recovered_near_misses:
        print("Near misses promovidos:")
        for result in recovered_near_misses:
            print(f"- {result['query']} | {result['expected_tool']}")
    else:
        print("Near misses promovidos: nenhum")

    if lost_existing_hits:
        print("Hits@2 perdidos:")
        for result in lost_existing_hits:
            print(
                f"- {result['query']} | {result['expected_tool']} | "
                f"Top-2 MMR={format_top_2(result['mmr_top_2'])}"
            )
    else:
        print("Hits@2 perdidos: nenhum")


if __name__ == "__main__":
    main()
