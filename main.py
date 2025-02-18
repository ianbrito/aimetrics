from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from ml.experiment import Experiment
from database.database import Database
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
exp_rf = Experiment(RandomForestClassifier(), dataset_name="Iris Dataset")
results_rf = exp_rf.train(X_iris, y_iris)
db.save_metrics(**results_rf)
print("RandomForestClassifier:", results_rf)

# LogisticRegression
exp_logreg = Experiment(LogisticRegression(max_iter=200), dataset_name="Iris Dataset")
results_logreg = exp_logreg.train(X_iris, y_iris)
db.save_metrics(**results_logreg)
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
db.save_metrics(**results_lr)
print("LinearRegression:", results_lr)

# RandomForestRegressor
exp_rfr = Experiment(RandomForestRegressor(), dataset_name="California Housing Dataset")
results_rfr = exp_rfr.train(X_housing, y_housing)
db.save_metrics(**results_rfr)
print("RandomForestRegressor:", results_rfr)

db.close()
