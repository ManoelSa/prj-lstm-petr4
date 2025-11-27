import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Erro Percentual Absoluto Médio (MAPE).
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.
        
    Returns:
        float: Erro percentual médio (%).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filtra índices onde o valor real não é zero para evitar divisão por zero
    non_zero_indices = y_true != 0
    if not np.any(non_zero_indices):
        return 0.0
    
    # Cálculo do MAPE, apenas nos índices não-zero
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

def create_sequences(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria o dataset estruturado em sequências (X) e alvos (Y) para LSTM.
    
    Args:
        data (np.ndarray): Série já escalonada, formato (n, 1).
        time_step (int): Número de passos de tempo usados como entrada.
        
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X: array 3D com shape (num_amostras, time_step, 1)
            - Y: array 1D com shape (num_amostras,)
    """
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Reshape para o formato 3D da LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, Y

def evaluate_predictions(
    Y_teste_original: np.ndarray,
    Y_previsao_original: np.ndarray
) -> dict[str, float]:
    """
    Calcula e retorna as métricas de avaliação do modelo.
    
    Args:
        Y_teste_original (np.ndarray): Valores reais (desnormalizados).
        Y_previsao_original (np.ndarray): Valores previstos (desnormalizados).
        
    Returns:
        dict[str, float]: Dicionário com métricas (RMSE, MAE, MAPE).
    
    """
    rmse = math.sqrt(mean_squared_error(Y_teste_original, Y_previsao_original))
    mae = mean_absolute_error(Y_teste_original, Y_previsao_original)
    mape = calculate_mape(Y_teste_original, Y_previsao_original)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }