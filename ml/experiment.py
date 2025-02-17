import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


class Experiment:
    def __init__(self, model, preprocessor=None, params=None, dataset_name=""):
        self.model = model.set_params(**(params or {}))
        self.preprocessor = preprocessor if preprocessor else StandardScaler()
        self.pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("model", self.model)]
        )
        self.dataset_name = dataset_name
        self.results = {}

    def train(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)

        # Escolher métricas de acordo com o tipo de problema
        if len(set(y)) < 10:  # Classificação
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }
        else:  # Regressão
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2_score": r2_score(y_test, y_pred),
            }

        self.results = {
            "dataset_name": self.dataset_name,
            "model_name": self.model.__class__.__name__,
            "params": self.model.get_params(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "metrics": metrics,
        }

        return self.results
    
    def cross_validate(self, X, y, cv=5):
        """Executa validação cruzada e retorna os scores."""
        scores = cross_val_score(self.pipeline, X, y, cv=cv)
        return {"cv_mean_score": np.mean(scores), "cv_scores": scores.tolist()}
