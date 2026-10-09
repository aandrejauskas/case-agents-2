"""Audita evidências objetivas sobre o ground truth das tools do eval dataset."""
import re
import unicodedata
from typing import Dict, List, Set

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from experiments.evaluate_candidate_union import tool_to_text
from common.data_loader import load_eval_dataset, load_tools


MODEL_ID = "intfloat/multilingual-e5-small"
STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "eu",
    "me",
    "meu",
    "minha",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "pra",
    "preciso",
    "quero",
    "que",
    "se",
    "um",
    "uma",
}
GROUND_TRUTH_NOTICE = (
    "O fato de outra tool possuir score maior ou nome mais específico não prova "
    "que a expected_tool está incorreta. O dataset é o ground truth oficial do "
    "case. Esta auditoria apenas identifica casos que merecem revisão humana."
)


def normalized_tokens(text: str) -> Set[str]:
    """Extrai tokens literais comparáveis, sem inferência semântica."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    tokens = re.findall(r"[a-z0-9]+", without_accents)
    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) >= 3
    }


def tool_name_tokens(tool_name: str) -> Set[str]:
    return normalized_tokens(tool_name.replace("_", " "))


def main() -> None:
    tools = load_tools()
    evaluation_items = [
        item for item in load_eval_dataset() if item.get("expected_tool")
    ]
    tool_names = [tool.name for tool in tools]
    tool_index_by_name = {name: index for index, name in enumerate(tool_names)}
    tool_texts = [f"passage: {tool_to_text(tool)}" for tool in tools]

    model = SentenceTransformer(MODEL_ID, device="cpu")
    tool_embeddings = model.encode(
        tool_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = []
    for item in evaluation_items:
        query = item["query"]
        expected_tool_name = item["expected_tool"]
        expected_index = tool_index_by_name[expected_tool_name]
        expected_tool = tools[expected_index]

        query_embedding = model.encode(
            [f"query: {query}"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = cosine_similarity(query_embedding, tool_embeddings)[0]
        order = np.argsort(-scores, kind="stable")
        expected_rank = int(np.flatnonzero(order == expected_index)[0]) + 1
        expected_score = float(scores[expected_index])

        query_tokens = normalized_tokens(query)
        expected_overlap = sorted(
            query_tokens & tool_name_tokens(expected_tool_name)
        )
        top_5 = []
        for rank, tool_index in enumerate(order[:5], start=1):
            tool = tools[int(tool_index)]
            overlap = sorted(query_tokens & tool_name_tokens(tool.name))
            top_5.append(
                {
                    "rank": rank,
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "score": float(scores[tool_index]),
                    "query_name_overlap": overlap,
                }
            )

        same_category_above = [
            tools[int(tool_index)].name
            for tool_index in order[: expected_rank - 1]
            if tools[int(tool_index)].category == expected_tool.category
        ]
        lexical_name_matches = []
        for rank, tool_index in enumerate(order, start=1):
            tool = tools[int(tool_index)]
            if tool.name == expected_tool_name:
                continue
            overlap = sorted(query_tokens & tool_name_tokens(tool.name))
            if overlap:
                lexical_name_matches.append(
                    {"rank": rank, "name": tool.name, "overlap": overlap}
                )

        clearer_top_5_overlaps = [
            tool
            for tool in top_5
            if tool["name"] != expected_tool_name
            and len(tool["query_name_overlap"]) > len(expected_overlap)
        ]

        results.append(
            {
                "query": query,
                "expected_tool": expected_tool_name,
                "expected_description": expected_tool.description,
                "expected_category": expected_tool.category,
                "expected_rank": expected_rank,
                "expected_score": expected_score,
                "expected_name_overlap": expected_overlap,
                "top_5": top_5,
                "top_1_gap": float(scores[order[0]]) - expected_score,
                "top_2_gap": float(scores[order[1]]) - expected_score,
                "in_top_1": expected_rank == 1,
                "in_top_2": expected_rank <= 2,
                "in_top_5": expected_rank <= 5,
                "in_top_10": expected_rank <= 10,
                "same_category_above": same_category_above,
                "lexical_name_matches": lexical_name_matches,
                "clearer_top_5_overlaps": clearer_top_5_overlaps,
            }
        )

    print_report(results)


def yes_no(value: bool) -> str:
    return "sim" if value else "não"


def print_name_list(items: List[str], limit: int = 8) -> str:
    if not items:
        return "nenhuma"
    visible = ", ".join(items[:limit])
    remaining = len(items) - limit
    return f"{visible} (+{remaining})" if remaining > 0 else visible


def print_report(results: List[dict]) -> None:
    print("AUDITORIA QUALITATIVA DO GROUND TRUTH DE TOOLS")
    print("=" * 112)
    print(f"Modelo de referência: {MODEL_ID}")
    print(f"Queries com expected_tool: {len(results)}")
    print(GROUND_TRUTH_NOTICE)
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. Query: {result['query']}")
        print(f"   expected_tool: {result['expected_tool']}")
        print(f"   name: {result['expected_tool']}")
        print(f"   description: {result['expected_description']}")
        print(f"   category: {result['expected_category']}")
        print(f"   rank E5 Small Dense: {result['expected_rank']}")
        print(f"   cosine similarity: {result['expected_score']:.6f}")
        print("   Top-5:")
        for tool in result["top_5"]:
            overlap = ", ".join(tool["query_name_overlap"]) or "nenhuma"
            print(
                f"      {tool['rank']}. {tool['name']} | "
                f"score={tool['score']:.6f} | category={tool['category']}"
            )
            print(f"         description: {tool['description']}")
            print(f"         palavras query↔name: {overlap}")
        print(
            "   diferenças de score "
            "(score da posição - score da expected): "
            f"Top-1={result['top_1_gap']:+.6f}; "
            f"Top-2={result['top_2_gap']:+.6f}"
        )
        print(
            "   posições: "
            f"Top-1={yes_no(result['in_top_1'])}; "
            f"Top-2={yes_no(result['in_top_2'])}; "
            f"Top-5={yes_no(result['in_top_5'])}; "
            f"Top-10={yes_no(result['in_top_10'])}"
        )
        same_category = result["same_category_above"]
        print(
            "   outras tools da mesma categoria acima: "
            f"{yes_no(bool(same_category))} ({len(same_category)})"
        )
        if same_category:
            print(f"      {print_name_list(same_category)}")
        lexical_matches = result["lexical_name_matches"]
        print(
            "   outras tools com palavras da query no nome: "
            f"{yes_no(bool(lexical_matches))} ({len(lexical_matches)})"
        )
        for match in lexical_matches[:8]:
            print(
                f"      rank {match['rank']}: {match['name']} | "
                f"palavras={', '.join(match['overlap'])}"
            )
        if len(lexical_matches) > 8:
            print(f"      ... e mais {len(lexical_matches) - 8}")
        print()

    print("FOCO ESPECIAL")
    print("=" * 112)
    print_rank_group("Expected abaixo do rank 2", results, 2)
    print_rank_group("Expected abaixo do rank 5", results, 5)
    print_rank_group("Expected abaixo do rank 10", results, 10)

    print("Alternativas Top-5 com mais sobreposição lexical que a expected:")
    lexical_cases = [
        result for result in results if result["clearer_top_5_overlaps"]
    ]
    if lexical_cases:
        for result in lexical_cases:
            expected_overlap = ", ".join(result["expected_name_overlap"]) or "nenhuma"
            print(
                f"- {result['query']} | expected={result['expected_tool']} "
                f"(palavras={expected_overlap})"
            )
            for tool in result["clearer_top_5_overlaps"]:
                print(
                    f"  rank {tool['rank']}: {tool['name']} | "
                    f"palavras={', '.join(tool['query_name_overlap'])}"
                )
    else:
        print("- nenhuma")
    print()

    print("RESUMO FINAL")
    print("=" * 112)
    print(
        "query | expected_tool | rank | expected_score | "
        "top1_tool | top1_score | top2_tool | top2_score | "
        "in_top2 | in_top5 | in_top10"
    )
    for result in results:
        top_1 = result["top_5"][0]
        top_2 = result["top_5"][1]
        print(
            f"{result['query']} | {result['expected_tool']} | "
            f"{result['expected_rank']} | {result['expected_score']:.6f} | "
            f"{top_1['name']} | {top_1['score']:.6f} | "
            f"{top_2['name']} | {top_2['score']:.6f} | "
            f"{yes_no(result['in_top_2'])} | "
            f"{yes_no(result['in_top_5'])} | "
            f"{yes_no(result['in_top_10'])}"
        )

    print()
    print("CONTAGENS")
    print(f"Expected Top-1: {sum(result['in_top_1'] for result in results)}")
    print(f"Expected Top-2: {sum(result['in_top_2'] for result in results)}")
    print(f"Expected Top-5: {sum(result['in_top_5'] for result in results)}")
    print(f"Expected Top-10: {sum(result['in_top_10'] for result in results)}")
    print(
        "Expected abaixo de Top-10: "
        f"{sum(not result['in_top_10'] for result in results)}"
    )
    print()
    print(GROUND_TRUTH_NOTICE)


def print_rank_group(title: str, results: List[dict], threshold: int) -> None:
    selected = [result for result in results if result["expected_rank"] > threshold]
    print(f"{title}: {len(selected)}")
    if selected:
        for result in selected:
            print(
                f"- rank {result['expected_rank']}: {result['query']} | "
                f"{result['expected_tool']}"
            )
    else:
        print("- nenhuma")


if __name__ == "__main__":
    main()
