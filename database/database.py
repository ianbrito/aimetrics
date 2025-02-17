import json
import psycopg2
import os


class Database:
    def __init__(self):
        dsn = f"dbname={os.getenv('DB_DATABASE')} user={os.getenv('DB_USERNAME')} password={os.getenv('DB_PASSWORD')} host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')}"
        print(dsn)
        self.conn = psycopg2.connect(dsn)

    def save_metrics(
        self, dataset_name, model_name, params, train_size, test_size, metrics
    ):
        """Salva o modelo e as métricas no banco de dados."""

        cursor = self.conn.cursor()

        stmt = """INSERT INTO models (model_name, params, train_size, test_size, dataset) VALUES (%s, %s, %s, %s, %s) RETURNING id;"""
        cursor.execute(
            stmt, (model_name, json.dumps(params), train_size, test_size, dataset_name)
        )

        model_id = cursor.fetchone()[0]

        for metric_name, metric_value in metrics.items():
            stmt = """INSERT INTO metrics (model_id, metric_name, metric_value) VALUES (%s, %s, %s);"""
            cursor.execute(stmt, (model_id, metric_name, metric_value))

        self.conn.commit()
        cursor.close()
        self.conn.close()
