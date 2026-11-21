"""
Service : EntityNormalizer
Normalisation des entités scientifiques extraites (méthodes, datasets, métriques).
Service concret — pas de LLM, règles déterministes.
"""

from __future__ import annotations


class EntityNormalizer:
    """
    Normalise les noms d'entités scientifiques pour éviter les doublons
    dans le Knowledge Graph Neo4j (Phase 6+).

    Ex: "RF", "random forest", "Random-Forest" → "Random Forest"
    """

    _METHOD_ALIASES: dict[str, str] = {
        "rf": "Random Forest",
        "random forest": "Random Forest",
        "random-forest": "Random Forest",
        "svm": "Support Vector Machine",
        "support vector machine": "Support Vector Machine",
        "support-vector machine": "Support Vector Machine",
        "lstm": "Long Short-Term Memory",
        "long short-term memory": "Long Short-Term Memory",
        "bert": "BERT",
        "cnn": "Convolutional Neural Network",
        "convolutional neural network": "Convolutional Neural Network",
        "rnn": "Recurrent Neural Network",
        "recurrent neural network": "Recurrent Neural Network",
        "xgboost": "XGBoost",
        "gradient boosting": "Gradient Boosting",
        "logistic regression": "Logistic Regression",
        "lr": "Logistic Regression",
        "knn": "K-Nearest Neighbors",
        "k-nearest neighbors": "K-Nearest Neighbors",
        "naive bayes": "Naive Bayes",
        "decision tree": "Decision Tree",
        "dt": "Decision Tree",
        "transformer": "Transformer",
        "gpt": "GPT",
        "llm": "Large Language Model",
        "large language model": "Large Language Model",
    }

    _METRIC_ALIASES: dict[str, str] = {
        "f1": "F1-score",
        "f1-score": "F1-score",
        "f1 score": "F1-score",
        "f-measure": "F1-score",
        "accuracy": "Accuracy",
        "acc": "Accuracy",
        "precision": "Precision",
        "prec": "Precision",
        "recall": "Recall",
        "sensitivity": "Recall",
        "auc": "AUC-ROC",
        "auc-roc": "AUC-ROC",
        "roc auc": "AUC-ROC",
        "mse": "Mean Squared Error",
        "mean squared error": "Mean Squared Error",
        "rmse": "Root Mean Squared Error",
        "root mean squared error": "Root Mean Squared Error",
        "mae": "Mean Absolute Error",
        "mean absolute error": "Mean Absolute Error",
        "bleu": "BLEU",
        "rouge": "ROUGE",
        "mrr": "Mean Reciprocal Rank",
        "ndcg": "nDCG",
        "map": "Mean Average Precision",
    }

    def normalize_method(self, name: str) -> str:
        """Normalise un nom de méthode ML/IA."""
        return self._METHOD_ALIASES.get(name.lower().strip(), name.strip())

    def normalize_metric(self, name: str) -> str:
        """Normalise un nom de métrique d'évaluation."""
        return self._METRIC_ALIASES.get(name.lower().strip(), name.strip())

    def normalize_dataset(self, name: str) -> str:
        """Normalise un nom de dataset (nettoyage basique)."""
        return name.strip()

    def normalize_concept(self, name: str) -> str:
        """Normalise un concept scientifique (lowercase + strip)."""
        return name.lower().strip()
