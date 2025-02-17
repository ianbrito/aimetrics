DROP TABLE IF EXISTS metrics;
DROP TABLE IF EXISTS models;

CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    train_size INT,
    test_size INT,
    params TEXT,
    dataset VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(id) ON DELETE CASCADE,
    metric_name VARCHAR(50),
    metric_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
