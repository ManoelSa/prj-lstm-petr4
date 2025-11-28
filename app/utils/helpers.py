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