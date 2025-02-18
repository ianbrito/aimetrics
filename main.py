from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from experiment import Experiment
from database import Database
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

db = Database()

# CLASSIFICAÇÃO
iris = load_iris()
iris_data_frame = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data_frame["target"] = iris.target

X_iris, y_iris = iris_data_frame[iris.feature_names], iris_data_frame["target"]

# RandomForestClassifier
exp_rf = Experiment(RandomForestClassifier(), dataset_name="Iris Dataset", sampling="oversampling")
results_rf = exp_rf.train(X_iris, y_iris)

model_id = db.save_model(results_rf["dataset_name"], results_rf["model_name"])
db.save_hyperparameters(model_id, results_rf["params"])
db.save_metrics(model_id, results_rf["metrics"])
db.save_preprocessing_strategy(model_id, "StandardScaler")
db.save_learning_strategy(model_id, "Hold-out")

print("RandomForestClassifier:", results_rf)

# LogisticRegression
exp_logreg = Experiment(LogisticRegression(max_iter=200), dataset_name="Iris Dataset", sampling="oversampling")
results_logreg = exp_logreg.train(X_iris, y_iris)

model_id = db.save_model(results_logreg["dataset_name"], results_logreg["model_name"])
db.save_hyperparameters(model_id, results_logreg["params"])
db.save_metrics(model_id, results_logreg["metrics"])
db.save_preprocessing_strategy(model_id, "StandardScaler")
db.save_learning_strategy(model_id, "Hold-out")

print("LogisticRegression:", results_logreg)

# REGRESSÃO
housing = fetch_california_housing()
housing_data_frame = pd.DataFrame(data=housing.data, columns=housing.feature_names)
housing_data_frame["target"] = housing.target

X_housing, y_housing = (
    housing_data_frame[housing.feature_names],
    housing_data_frame["target"],
)

# LinearRegression
exp_lr = Experiment(LinearRegression(), dataset_name="California Housing Dataset")
results_lr = exp_lr.train(X_housing, y_housing)

model_id = db.save_model(results_lr["dataset_name"], results_lr["model_name"])
db.save_hyperparameters(model_id, results_lr["params"])
db.save_metrics(model_id, results_lr["metrics"])
db.save_preprocessing_strategy(model_id, "StandardScaler")
db.save_learning_strategy(model_id, "Hold-out")

print("LinearRegression:", results_lr)

# RandomForestRegressor
exp_rfr = Experiment(RandomForestRegressor(), dataset_name="California Housing Dataset")
results_rfr = exp_rfr.train(X_housing, y_housing)

model_id = db.save_model(results_rfr["dataset_name"], results_rfr["model_name"])
db.save_hyperparameters(model_id, results_rfr["params"])
db.save_metrics(model_id, results_rfr["metrics"])
db.save_preprocessing_strategy(model_id, "StandardScaler")
db.save_learning_strategy(model_id, "Hold-out")


print("RandomForestRegressor:", results_rfr)

db.close()
