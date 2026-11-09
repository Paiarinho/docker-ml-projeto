# Aqui desenvolveremos o proesso de treino
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sqlalchemy import create_engine

from utils.logger import get_logger

logger = get_logger(__name__)
# Passo 1
# FUncao para leitura do dataset
def read_dataset(path):
    logger.info("Carregando o dataset...")
    engine = create_engine(f"sqlite:///data/{path}")
    df = pd.read_sql_query(
        "SELECT timestamp, heat_efficiency FROM heat_exchanger ORDER BY timestamp",
        engine,
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_index'] = (df['timestamp'] - df['timestamp'].min()).dt.days
    return df

# Funcao para treinar o modelo
def train(x:np.ndarray, y:np.ndarray) -> LinearRegression:
    logger.info('Treinando o Modelo...')
    model = LinearRegression()
    model.fit(x, y)
    return model

# Funcao para salvar o modelo treinado
def save_model(model: LinearRegression, path:str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

# Funcao para avaliar o modelo
def evaluate(model:LinearRegression, y:np.ndarray, x:np.ndarray):
    pred = model.predict(x)
    r2 = r2_score(y, pred)
    logger.info(f"R² Score: {r2:.4f}")
    return r2

if __name__ == "__main__":
    # Carregar os dados
    df = read_dataset(path='heat_exchanger.db')

    # Treinar o modelo
    x = df['day_index'].values.reshape(-1, 1)
    y = df['heat_efficiency'].values
    model = train(x, y)

    # Avaliar modelo
    evaluate(model, y, x)

    # Salvar o modelo treinado
    save_model(model, path='artifacts/heat_efficiency_model.pkl')