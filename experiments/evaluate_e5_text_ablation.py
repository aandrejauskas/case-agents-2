"""Ablação diagnóstica da representação textual das tools no E5 Small.

Compara somente as três representações solicitadas, sem alterar ou importar o
Retriever de produção. O eval já foi reutilizado e os resultados deste script
não devem ser tratados como evidência de generalização.
"""
from statistics import median
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from common.data_loader import load_eval_dataset, load_tools
from common.schemas import Tool


MODEL_ID = "intfloat/multilingual-e5-small"
VARIANT_NAMES = ["A_CURRENT", "B_NO_CATEGORY", "C_DESCRIPTION_ONLY"]
TOP_K_VALUES = [1, 2, 3, 5, 10]


def tool_text(tool: Tool, variant: str) -> str:
    """Monta uma das três representações, sem pesos ou campos adicionais."""
    readable_name = tool.name.replace("_", " ")
    if variant == "A_CURRENT":
        return f"{readable_name}. {tool.description} Categoria: {tool.category}."
    if variant == "B_NO_CATEGORY":
        return f"{readable_name}. {tool.description}"
    if variant == "C_DESCRIPTION_ONLY":
        return tool.description
    raise ValueError(f"Variante desconhecida: {variant}")


def calculate_metrics(ranks: List[int]) -> Dict[str, float]:
    total = len(ranks)
    metrics = {
        f"Hit@{k}": sum(rank <= k for rank in ranks) / total
        for k in TOP_K_VALUES
    }
    metrics["MRR"] = sum(1.0 / rank for rank in ranks) / total
    metrics["median_rank"] = float(median(ranks))
    return metrics


def calculate_ranks(
    query_embeddings: np.ndarray,
    tool_embeddings: np.ndarray,
    expected_indices: List[int],
) -> List[int]:
    similarities = cosine_similarity(query_embeddings, tool_embeddings)
    ranks = []
    for scores, expected_index in zip(similarities, expected_indices):
        order = np.argsort(-scores, kind="stable")
        ranks.append(int(np.flatnonzero(order == expected_index)[0]) + 1)
    return ranks


def top_2_changes(
    items: List[dict], baseline_ranks: List[int], candidate_ranks: List[int]
) -> Dict[str, List[dict]]:
    entries = []
    exits = []
    for item, old_rank, new_rank in zip(items, baseline_ranks, candidate_ranks):
        change = {
            "query": item["query"],
            "expected_tool": item["expected_tool"],
            "old_rank": old_rank,
            "new_rank": new_rank,
        }
        if old_rank > 2 and new_rank <= 2:
            entries.append(change)
        elif old_rank <= 2 and new_rank > 2:
            exits.append(change)
    return {"entries": entries, "exits": exits}


def print_change_list(title: str, changes: List[dict]) -> None:
    print(title)
    if not changes:
        print("- nenhuma")
        return
    for change in changes:
        print(
            f"- {change['query']} | {change['expected_tool']} | "
            f"{change['old_rank']} -> {change['new_rank']}"
        )


def is_focus_query(query: str) -> bool:
    normalized = query.casefold()
    markers = [
        "quanto eu tenho disponível",
        "cep",
        "linha digitável",
        "saldo disponível pra pix",
        "limite que ainda tenho",
        "pagamento da fatura em parcelas",
        "parcelar minha fatura",
        "aplicativo está travando",
    ]
    return any(marker in normalized for marker in markers)


def print_report(
    items: List[dict], ranks_by_variant: Dict[str, List[int]]
) -> None:
    metrics_by_variant = {
        variant: calculate_metrics(ranks)
        for variant, ranks in ranks_by_variant.items()
    }

    print("ABLAÇÃO DIAGNÓSTICA — REPRESENTAÇÃO TEXTUAL DAS TOOLS")
    print("=" * 118)
    print(f"Modelo: {MODEL_ID}")
    print("Configuração: query:/passage:, embeddings normalizados, cosine similarity")
    print(f"Catálogo: 285 tools | Queries com expected_tool: {len(items)}")
    print(
        "AVISO: o eval já foi reutilizado. Este resultado é somente diagnóstico; "
        "não é um holdout cego nem evidência suficiente de generalização."
    )
    print()

    print("MÉTRICAS")
    print("=" * 118)
    print(f"{'Métrica':<16}{'A — CURRENT':>20}{'B — NO_CATEGORY':>20}{'C — DESCRIPTION_ONLY':>24}")
    for metric_name in ["Hit@1", "Hit@2", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        print(
            f"{metric_name:<16}"
            f"{metrics_by_variant['A_CURRENT'][metric_name]:>20.6f}"
            f"{metrics_by_variant['B_NO_CATEGORY'][metric_name]:>20.6f}"
            f"{metrics_by_variant['C_DESCRIPTION_ONLY'][metric_name]:>24.6f}"
        )
    print(
        f"{'Mediana rank':<16}"
        f"{metrics_by_variant['A_CURRENT']['median_rank']:>20.1f}"
        f"{metrics_by_variant['B_NO_CATEGORY']['median_rank']:>20.1f}"
        f"{metrics_by_variant['C_DESCRIPTION_ONLY']['median_rank']:>24.1f}"
    )
    print()

    print("RANKS POR QUERY")
    print("=" * 118)
    print("query | expected_tool | rank_A | rank_B | rank_C")
    for index, item in enumerate(items):
        print(
            f"{item['query']} | {item['expected_tool']} | "
            f"{ranks_by_variant['A_CURRENT'][index]} | "
            f"{ranks_by_variant['B_NO_CATEGORY'][index]} | "
            f"{ranks_by_variant['C_DESCRIPTION_ONLY'][index]}"
        )
    print()

    print("MUDANÇAS NO TOP-2 EM RELAÇÃO A A — CURRENT")
    print("=" * 118)
    for variant, label in [
        ("B_NO_CATEGORY", "B — NO_CATEGORY"),
        ("C_DESCRIPTION_ONLY", "C — DESCRIPTION_ONLY"),
    ]:
        changes = top_2_changes(
            items,
            ranks_by_variant["A_CURRENT"],
            ranks_by_variant[variant],
        )
        print(label)
        print_change_list("Entradas novas no Top-2:", changes["entries"])
        print_change_list("Saídas do Top-2:", changes["exits"])
        print(
            "Variação líquida: "
            f"{len(changes['entries']) - len(changes['exits']):+d} query(s)"
        )
        print()

    print("QUERIES EM DESTAQUE")
    print("=" * 118)
    print("query | expected_tool | rank_A | rank_B | rank_C")
    for index, item in enumerate(items):
        if is_focus_query(item["query"]):
            print(
                f"{item['query']} | {item['expected_tool']} | "
                f"{ranks_by_variant['A_CURRENT'][index]} | "
                f"{ranks_by_variant['B_NO_CATEGORY'][index]} | "
                f"{ranks_by_variant['C_DESCRIPTION_ONLY'][index]}"
            )


def main() -> None:
    tools = load_tools()
    items = [item for item in load_eval_dataset() if item.get("expected_tool")]
    tool_index_by_name = {tool.name: index for index, tool in enumerate(tools)}
    expected_indices = [tool_index_by_name[item["expected_tool"]] for item in items]

    model = SentenceTransformer(MODEL_ID, device="cpu")
    query_embeddings = model.encode(
        [f"query: {item['query']}" for item in items],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    ranks_by_variant = {}
    for variant in VARIANT_NAMES:
        tool_embeddings = model.encode(
            [f"passage: {tool_text(tool, variant)}" for tool in tools],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        ranks_by_variant[variant] = calculate_ranks(
            query_embeddings, tool_embeddings, expected_indices
        )

    print_report(items, ranks_by_variant)


if __name__ == "__main__":
    main()
