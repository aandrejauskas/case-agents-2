"""Compara o Router MiniLM atual com uma baseline lexical TF-IDF.

Experimento isolado: não é importado pelo runtime e não modifica datasets ou
componentes de produção. O eval já foi observado e serve apenas como diagnóstico.
"""
from statistics import median
from time import perf_counter
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from candidate_starter.embeddings import TextEmbedder
from common.data_loader import load_eval_dataset, load_router_training_data


LABELS = ["FAST_PATH", "AGENT"]
MINILM_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TFIDF_OPTIONS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "norm": "l2",
}
LOGREG_OPTIONS = {"random_state": 42, "max_iter": 1_000}
SALDO_PIX_QUERY = "Preciso saber o saldo disponível pra pix"


def milliseconds(start: float) -> float:
    return (perf_counter() - start) * 1_000


def latency_summary(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def evaluate_minilm(
    train_texts: List[str], train_labels: List[str], queries: List[str]
) -> dict:
    initialization_start = perf_counter()
    embedder = TextEmbedder(MINILM_MODEL_ID)
    classifier = LogisticRegression(**LOGREG_OPTIONS)
    initialization_ms = milliseconds(initialization_start)

    fit_start = perf_counter()
    train_features = embedder.embed(train_texts)
    classifier.fit(train_features, train_labels)
    fit_ms = milliseconds(fit_start)

    predictions = []
    total_latencies = []
    transform_latencies = []
    classifier_latencies = []
    for query in queries:
        total_start = perf_counter()

        transform_start = perf_counter()
        query_features = embedder.embed([query])
        transform_latencies.append(milliseconds(transform_start))

        classifier_start = perf_counter()
        probabilities = classifier.predict_proba(query_features)[0]
        predicted_index = int(np.argmax(probabilities))
        predictions.append(str(classifier.classes_[predicted_index]))
        classifier_latencies.append(milliseconds(classifier_start))

        total_latencies.append(milliseconds(total_start))

    return {
        "name": "A — MINILM_LOGREG",
        "predictions": predictions,
        "initialization_ms": initialization_ms,
        "fit_ms": fit_ms,
        "predict_latency": latency_summary(total_latencies),
        "transform_latency": latency_summary(transform_latencies),
        "classifier_latency": latency_summary(classifier_latencies),
    }


def evaluate_tfidf(
    train_texts: List[str], train_labels: List[str], queries: List[str]
) -> dict:
    initialization_start = perf_counter()
    vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    classifier = LogisticRegression(**LOGREG_OPTIONS)
    initialization_ms = milliseconds(initialization_start)

    fit_start = perf_counter()
    train_features = vectorizer.fit_transform(train_texts)
    classifier.fit(train_features, train_labels)
    fit_ms = milliseconds(fit_start)

    predictions = []
    total_latencies = []
    transform_latencies = []
    classifier_latencies = []
    for query in queries:
        total_start = perf_counter()

        transform_start = perf_counter()
        query_features = vectorizer.transform([query])
        transform_latencies.append(milliseconds(transform_start))

        classifier_start = perf_counter()
        probabilities = classifier.predict_proba(query_features)[0]
        predicted_index = int(np.argmax(probabilities))
        predictions.append(str(classifier.classes_[predicted_index]))
        classifier_latencies.append(milliseconds(classifier_start))

        total_latencies.append(milliseconds(total_start))

    return {
        "name": "B — TFIDF_LOGREG",
        "predictions": predictions,
        "initialization_ms": initialization_ms,
        "fit_ms": fit_ms,
        "predict_latency": latency_summary(total_latencies),
        "transform_latency": latency_summary(transform_latencies),
        "classifier_latency": latency_summary(classifier_latencies),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "training_shape": train_features.shape,
    }


def classification_metrics(y_true: List[str], y_pred: List[str]) -> dict:
    confusion = {
        actual: {predicted: 0 for predicted in LABELS} for actual in LABELS
    }
    for actual, predicted in zip(y_true, y_pred):
        confusion[actual][predicted] += 1

    true_positive = confusion["AGENT"]["AGENT"]
    false_positive = confusion["FAST_PATH"]["AGENT"]
    false_negative = confusion["AGENT"]["FAST_PATH"]
    precision_agent = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall_agent = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1_agent = (
        2 * precision_agent * recall_agent / (precision_agent + recall_agent)
        if precision_agent + recall_agent
        else 0.0
    )
    accuracy = sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)

    return {
        "accuracy": accuracy,
        "precision_agent": precision_agent,
        "recall_agent": recall_agent,
        "f1_agent": f1_agent,
        "confusion_matrix": confusion,
        "fast_path_to_agent": false_positive,
        "agent_to_fast_path": false_negative,
    }


def print_confusion(name: str, confusion: dict) -> None:
    print(name)
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


def print_latency(label: str, values: dict) -> None:
    print(
        f"{label:<26} média={values['mean']:.4f} ms | "
        f"mediana={values['median']:.4f} ms | p95={values['p95']:.4f} ms"
    )


def print_change_list(title: str, changes: List[dict]) -> None:
    print(title)
    if not changes:
        print("- nenhuma")
        return
    for change in changes:
        print(
            f"- {change['query']} | expected={change['expected_route']} | "
            f"MiniLM={change['minilm_prediction']} | "
            f"TF-IDF={change['tfidf_prediction']}"
        )


def print_report(eval_items: List[dict], minilm: dict, tfidf: dict) -> None:
    y_true = [item["expected_route"] for item in eval_items]
    minilm_metrics = classification_metrics(y_true, minilm["predictions"])
    tfidf_metrics = classification_metrics(y_true, tfidf["predictions"])

    print("ABLAÇÃO DIAGNÓSTICA DO ROUTER — MINILM × TF-IDF")
    print("=" * 100)
    print(f"Treino: data/router_training_data.json | Avaliação: {len(eval_items)} queries")
    print(
        "AVISO: o eval já foi observado. O resultado é diagnóstico comparativo, "
        "não evidência forte de generalização; 1 query equivale a 3,33 p.p."
    )
    print()

    print("TABELA COMPARATIVA")
    print("=" * 100)
    print(f"{'Métrica':<34}{'A — MINILM':>20}{'B — TF-IDF':>20}")
    rows = [
        ("Accuracy", "accuracy", "%"),
        ("Precision AGENT", "precision_agent", "%"),
        ("Recall AGENT", "recall_agent", "%"),
        ("F1 AGENT", "f1_agent", "%"),
    ]
    for label, key, _ in rows:
        print(
            f"{label:<34}{minilm_metrics[key]:>19.2%}{tfidf_metrics[key]:>20.2%}"
        )
    print(
        f"{'FAST_PATH real → AGENT':<34}"
        f"{minilm_metrics['fast_path_to_agent']:>20}"
        f"{tfidf_metrics['fast_path_to_agent']:>20}"
    )
    print(
        f"{'AGENT real → FAST_PATH':<34}"
        f"{minilm_metrics['agent_to_fast_path']:>20}"
        f"{tfidf_metrics['agent_to_fast_path']:>20}"
    )
    print()

    print("CONFUSION MATRICES")
    print("=" * 100)
    print_confusion("A — MINILM_LOGREG", minilm_metrics["confusion_matrix"])
    print()
    print_confusion("B — TFIDF_LOGREG", tfidf_metrics["confusion_matrix"])
    print()

    print("TEMPOS")
    print("=" * 100)
    for experiment in [minilm, tfidf]:
        print(experiment["name"])
        print(f"Inicialização: {experiment['initialization_ms']:.4f} ms")
        print(f"Fit total: {experiment['fit_ms']:.4f} ms")
        print_latency("Predict total", experiment["predict_latency"])
        print_latency("Transformação da query", experiment["transform_latency"])
        print_latency("Classificador", experiment["classifier_latency"])
        if "vocabulary_size" in experiment:
            print(f"Vocabulário TF-IDF: {experiment['vocabulary_size']}")
            print(f"Shape de treino: {experiment['training_shape']}")
        print()

    divergences = []
    corrected = []
    introduced = []
    for item, minilm_prediction, tfidf_prediction in zip(
        eval_items, minilm["predictions"], tfidf["predictions"]
    ):
        if minilm_prediction == tfidf_prediction:
            continue
        difference = {
            "query": item["query"],
            "expected_route": item["expected_route"],
            "minilm_prediction": minilm_prediction,
            "tfidf_prediction": tfidf_prediction,
        }
        divergences.append(difference)
        if minilm_prediction != item["expected_route"] and tfidf_prediction == item["expected_route"]:
            corrected.append(difference)
        elif minilm_prediction == item["expected_route"] and tfidf_prediction != item["expected_route"]:
            introduced.append(difference)

    print("QUERIES EM QUE A E B DISCORDAM")
    print("=" * 100)
    print_change_list("Divergências:", divergences)
    print()
    print_change_list("Erros do MiniLM corrigidos pelo TF-IDF:", corrected)
    print()
    print_change_list("Erros novos introduzidos pelo TF-IDF:", introduced)
    print()

    saldo_pix_index = next(
        index for index, item in enumerate(eval_items) if item["query"] == SALDO_PIX_QUERY
    )
    print("FALSO NEGATIVO DE SALDO PIX")
    print("=" * 100)
    print(f"Query: {SALDO_PIX_QUERY}")
    print(f"Expected: {eval_items[saldo_pix_index]['expected_route']}")
    print(f"MiniLM: {minilm['predictions'][saldo_pix_index]}")
    print(f"TF-IDF: {tfidf['predictions'][saldo_pix_index]}")


def main() -> None:
    train_texts, train_labels = load_router_training_data()
    eval_items = load_eval_dataset()
    queries = [item["query"] for item in eval_items]

    minilm = evaluate_minilm(train_texts, train_labels, queries)
    tfidf = evaluate_tfidf(train_texts, train_labels, queries)
    print_report(eval_items, minilm, tfidf)


if __name__ == "__main__":
    main()
