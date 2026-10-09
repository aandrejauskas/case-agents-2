"""Avalia somente a classificação de rotas do QueryRouter atual."""
from typing import Dict, List

from candidate_starter.router import QueryRouter
from common.data_loader import load_eval_dataset, load_router_training_data


LABELS = ["FAST_PATH", "AGENT"]


def main() -> None:
    train_texts, train_labels = load_router_training_data()
    eval_dataset = load_eval_dataset()
    router = QueryRouter().fit(train_texts, train_labels)

    confusion: Dict[str, Dict[str, int]] = {
        actual: {predicted: 0 for predicted in LABELS} for actual in LABELS
    }
    errors: List[dict] = []

    for item in eval_dataset:
        result = router.predict(item["query"])
        expected_route = item["expected_route"]
        confusion[expected_route][result.route] += 1

        if result.route != expected_route:
            errors.append(
                {
                    "query": item["query"],
                    "expected_route": expected_route,
                    "predicted_route": result.route,
                    "confidence": result.confidence,
                }
            )

    total = len(eval_dataset)
    error_count = len(errors)
    correct_count = total - error_count
    accuracy = correct_count / total if total else 0.0

    print("AVALIAÇÃO DO ROUTER")
    print("=" * 72)
    print(f"Total de exemplos: {total}")
    print(f"Acertos: {correct_count}")
    print(f"Erros: {error_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print()
    print("CONFUSION MATRIX")
    print(f"{'':22}PREDITO")
    print(f"{'':18}{'FAST_PATH':>12}{'AGENT':>10}")
    print(
        f"{'REAL FAST_PATH':18}"
        f"{confusion['FAST_PATH']['FAST_PATH']:>12}"
        f"{confusion['FAST_PATH']['AGENT']:>10}"
    )
    print(
        f"{'REAL AGENT':18}"
        f"{confusion['AGENT']['FAST_PATH']:>12}"
        f"{confusion['AGENT']['AGENT']:>10}"
    )

    print_errors(
        "AGENT classificados como FAST_PATH",
        [error for error in errors if error["expected_route"] == "AGENT"],
    )
    print_errors(
        "FAST_PATH classificados como AGENT",
        [error for error in errors if error["expected_route"] == "FAST_PATH"],
    )


def print_errors(title: str, errors: List[dict]) -> None:
    print()
    print(title)
    print("-" * 72)
    if not errors:
        print("Nenhum erro.")
        return

    for index, error in enumerate(errors, start=1):
        confidence = error["confidence"]
        confidence_text = "N/A" if confidence is None else f"{confidence:.4f}"
        print(f"{index}. Query: {error['query']}")
        print(f"   expected_route: {error['expected_route']}")
        print(f"   predicted_route: {error['predicted_route']}")
        print(f"   confidence: {confidence_text}")


if __name__ == "__main__":
    main()
