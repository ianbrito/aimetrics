DROP TABLE IF EXISTS metrics;
DROP TABLE IF EXISTS model_preprocessing;
DROP TABLE IF EXISTS preprocessing_strategies;
DROP TABLE IF EXISTS model_learning;
DROP TABLE IF EXISTS learning_strategies;
DROP TABLE IF EXISTS model_hyperparameters;
DROP TABLE IF EXISTS models;

CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    dataset VARCHAR(255) NOT NULL
);

CREATE TABLE model_hyperparameters (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(id) ON DELETE CASCADE,
    param_name VARCHAR(255) NOT NULL,
    param_value TEXT NOT NULL
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL
);

CREATE TABLE preprocessing_strategies (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE learning_strategies (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE model_preprocessing (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(id) ON DELETE CASCADE,
    preprocessing_id INT REFERENCES preprocessing_strategies(id) ON DELETE CASCADE
);

CREATE TABLE model_learning (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(id) ON DELETE CASCADE,
    learning_id INT REFERENCES learning_strategies(id) ON DELETE CASCADE
);
