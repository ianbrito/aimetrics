import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class Experiment:
    def __init__(self, model, preprocessor=None, params=None, dataset_name="", sampling=None, learning_strategy="Hold-out"):
        self.model = model.set_params(**(params or {}))
        self.preprocessor = preprocessor if preprocessor else StandardScaler()
        self.pipeline = Pipeline([("preprocessor", self.preprocessor), ("model", self.model)])
        self.dataset_name = dataset_name
        self.sampling = sampling
        self.learning_strategy = learning_strategy
        self.results = {}

    def apply_sampling(self, X, y):
        """Aplica técnicas de amostragem."""
        if self.sampling == "oversampling":
            sampler = RandomOverSampler(random_state=42)
        elif self.sampling == "undersampling":
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X, y  # Sem amostragem
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def train(self, X, y, test_size=0.2):
        """Treina o modelo e registra as métricas."""
        X, y = self.apply_sampling(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        if pd.api.types.is_numeric_dtype(y):  # Regressão
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2_score": r2_score(y_test, y_pred),
            }
        else:  # Classificação
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }

        self.results = {
            "dataset_name": self.dataset_name,
            "model_name": self.model.__class__.__name__,
            "params": self.model.get_params(),
            "metrics": metrics,
        }

        return self.results
