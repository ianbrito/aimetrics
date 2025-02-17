from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from ml.experiment import Experiment
from database.database import Database
from sklearn.linear_model import LinearRegression
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

db = Database()

iris = load_iris()
iris_data_frame = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data_frame["target"] = iris.target
iris_data_frame["target_name"] = iris_data_frame["target"].apply(
    lambda x: iris.target_names[x]
)

X, y = iris_data_frame[iris.feature_names], iris_data_frame["target"]

# Criando experimento com RandomForest para classificação
exp = Experiment(RandomForestClassifier(), dataset_name="Iris Dataset")

results = exp.train(X, y)

db.save_metrics(
    results["dataset_name"],
    results["model_name"],
    results["params"],
    results["train_size"],
    results["test_size"],
    results["metrics"],
)

print("Resultados:", results)



exp_reg = Experiment(LinearRegression(), dataset_name="Boston Housing Dataset")

housing = fetch_california_housing()

housing_data_frame = pd.DataFrame(data=housing.data, columns=housing.feature_names)
housing_data_frame["target"] = housing.target 


results = exp_reg.train(X, y)

db.save_metrics(
    results["dataset_name"],
    results["model_name"],
    results["params"],
    results["train_size"],
    results["test_size"],
    results["metrics"],
)

print("Resultados:", results)