import os
from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{os.getenv("DB_USERNAME")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_DATABASE")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo de dados para as tabelas do banco
class Model(db.Model):
    __tablename__ = 'models'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(255), nullable=False)
    dataset = db.Column(db.String(255), nullable=False)
    metrics = db.relationship('Metric', backref='models', lazy=True)

class Metric(db.Model):
    __tablename__ = 'metrics'  # Use o nome correto da tabela 'metrics', no plural
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('models.id'), nullable=False)
    metric_name = db.Column(db.String(255), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)

    # Definir relacionamento com o modelo "models"
    model = db.relationship('Model', back_populates='metrics')

@app.route('/')
def index():
    # Consulta as métricas de todos os modelos
    metrics = Metric.query.join(Model).all()
    data = []
    for metric in metrics:
        data.append({
            'model_name': metric.model.model_name,
            'dataset': metric.model.dataset,
            'metric_name': metric.metric_name,
            'metric_value': metric.metric_value
        })

    # Cria um DataFrame com os dados
    df = pd.DataFrame(data)

    # Gráfico de comparação das métricas
    fig = px.bar(df, x="model_name", y="metric_value", color="metric_name", barmode="group", title="Comparação de Métricas por Modelo")

    # Converte o gráfico para HTML
    graph_html = fig.to_html(full_html=False)

    return render_template('index.html', graph_html=graph_html)

@app.route('/api/metrics')
def get_metrics():
    # Retorna os dados das métricas como JSON para a API
    metrics = Metric.query.join(Model).all()
    data = []
    for metric in metrics:
        data.append({
            'model_name': metric.model.model_name,
            'dataset': metric.model.dataset,
            'metric_name': metric.metric_name,
            'metric_value': metric.metric_value
        })
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
