import json
import psycopg2
import os


class Database:
    def __init__(self):
        dsn = f"dbname={os.getenv('DB_DATABASE')} user={os.getenv('DB_USERNAME')} password={os.getenv('DB_PASSWORD')} host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')}"
        self.conn = psycopg2.connect(dsn)
        self.cursor = self.conn.cursor()

    def save_model(self, dataset_name, model_name):
        """Salva um modelo no banco e retorna o ID."""
        stmt = """INSERT INTO models (model_name, dataset) VALUES (%s, %s) RETURNING id;"""
        self.cursor.execute(stmt, (model_name, dataset_name))
        model_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return model_id

    def save_hyperparameters(self, model_id, params):
        """Salva hiperparâmetros do modelo no banco."""
        stmt = """INSERT INTO model_hyperparameters (model_id, param_name, param_value) VALUES (%s, %s, %s);"""
        for param_name, param_value in params.items():
            self.cursor.execute(stmt, (model_id, param_name, json.dumps(param_value)))
        self.conn.commit()

    def save_metrics(self, model_id, metrics):
        """Salva métricas de desempenho do modelo no banco."""
        stmt = """INSERT INTO metrics (model_id, metric_name, metric_value) VALUES (%s, %s, %s);"""
        for metric_name, metric_value in metrics.items():
            self.cursor.execute(stmt, (model_id, metric_name, float(metric_value)))
        self.conn.commit()

    def save_preprocessing_strategy(self, model_id, strategy_name):
        """Registra a estratégia de pré-processamento usada no modelo."""
        self.cursor.execute(
            """INSERT INTO preprocessing_strategies (strategy_name) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id;""",
            (strategy_name,),
        )
        strategy_id = self.cursor.fetchone()
        if strategy_id:
            self.cursor.execute(
                """INSERT INTO model_preprocessing (model_id, preprocessing_id) VALUES (%s, %s);""",
                (model_id, strategy_id[0]),
            )
            self.conn.commit()

    def save_learning_strategy(self, model_id, strategy_name):
        """Registra a estratégia de aprendizado usada no modelo."""
        self.cursor.execute(
            """INSERT INTO learning_strategies (strategy_name) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id;""",
            (strategy_name,),
        )
        strategy_id = self.cursor.fetchone()
        if strategy_id:
            self.cursor.execute(
                """INSERT INTO model_learning (model_id, learning_id) VALUES (%s, %s);""",
                (model_id, strategy_id[0]),
            )
            self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()
