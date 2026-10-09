"""Router TF-IDF + Logistic Regression para FAST_PATH ou AGENT."""
import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common.interfaces import BaseRouter
from common.schemas import RouteResult


class QueryRouter(BaseRouter):
    def __init__(self) -> None:
        self._fitted = False
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        self.classifier = LogisticRegression(random_state=42, max_iter=1_000)
        self.training_shape = None

    def fit(self, texts: List[str], labels: List[str]) -> "QueryRouter":
        """Treina o router com os exemplos de `data/router_training_data.json`."""
        if not texts:
            raise ValueError("Informe pelo menos um texto para treinar o Router.")
        if len(texts) != len(labels):
            raise ValueError("texts e labels devem ter a mesma quantidade de itens.")
        if set(labels) != {"FAST_PATH", "AGENT"}:
            raise ValueError("Os dados de treino devem conter as classes FAST_PATH e AGENT.")

        # O mesmo vocabulário aprendido no fit transforma as queries no predict.
        features = self.vectorizer.fit_transform(texts)
        # Formato: (quantidade de exemplos, termos/ngrams do vocabulário).
        self.training_shape = features.shape
        self.classifier.fit(features, labels)
        self._fitted = True
        return self

    def predict(self, query: str) -> RouteResult:
        """Classifica `query` e retorna a rota escolhida + latência medida (em ms)."""
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")

        start = time.perf_counter()
        query_features = self.vectorizer.transform([query])

        # predict retornaria apenas a classe. predict_proba retorna a probabilidade
        # de cada classe; o maior valor define a rota e também vira a confiança.
        probabilities = self.classifier.predict_proba(query_features)[0]
        predicted_index = int(np.argmax(probabilities))
        route = str(self.classifier.classes_[predicted_index])
        confidence = float(probabilities[predicted_index])
        latency_ms = (time.perf_counter() - start) * 1_000

        return RouteResult(
            route=route,
            latency_ms=latency_ms,
            confidence=confidence,
        )
