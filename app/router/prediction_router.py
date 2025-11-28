import torch
import time
import logging
import yfinance as yf
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.prediction_schema import PredictionResponse 
from datetime import datetime, timedelta
from app.config.settings import TIME_STEP, TICKER 
from app.config.security import verify_token

# Importa o módulo de estado que contém as variáveis globais carregadas (MODEL, SCALER)
from app import state 

logger = logging.getLogger(__name__)

router = APIRouter()

# --- REQUISITO 5: Métricas ---

def get_ml_artifacts():
    """
    Dependência para injetar o Modelo e o Scaler no endpoint.
    Verifica o estado do modelo carregado no hook lifespan.
    """
   
    # Verifica o estado do módulo
    if state.MODEL is None or state.SCALER is None:
        raise HTTPException(status_code=503, detail="Serviço indisponível. Modelo ou Scaler não carregados.")
    
    # Retorna o modelo e o scaler carregados (apenas se não forem None)
    return state.MODEL, state.SCALER

@router.post("/predict/petr4", response_model=PredictionResponse)
def predict_price(artifacts: tuple = Depends(get_ml_artifacts), token: str = Depends(verify_token)):
    """
    Endpoint responsável por prever o preço de fechamento do próximo pregão (D+1)
    da ação PETR4.SA. A previsão utiliza o modelo LSTM treinado e a janela temporal
    de 60 dias.
    """
    start_time = time.time()
    
    # 1. Obtenção de Artefatos (Injetados)
    model, scaler = artifacts
    
    # 2. Busca de Dados Históricos (yfinance)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(TICKER, start=start_date, end=end_date)
        
        if data.empty or len(data) < TIME_STEP:
            raise HTTPException(
                status_code=400,
                detail=f"Não foi possível obter os últimos {TIME_STEP} preços de fechamento (Close) para {TICKER}."
            )
            
        recent_prices = data['Close'].tail(TIME_STEP).values
        ultima_data = data.index[-1].strftime("%Y-%m-%d")
        
        if len(recent_prices) != TIME_STEP:
            raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes. Encontrados apenas {len(recent_prices)} dias úteis, mas {TIME_STEP} são necessários."
            )

    except Exception as e:
        logger.error(f"Falha ao buscar dados históricos via yfinance para {TICKER}. Erro: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Falha ao buscar dados históricos via yfinance para {TICKER}. Erro interno."
        )

    # 3. Pré-processamento e Inferência (PyTorch)
    
    # Converte para NumPy (N, 1) para o scaler
    input_data_np = recent_prices.reshape(-1, 1)
    scaled_input_data = scaler.transform(input_data_np)
    
    # Converte para Tensor (1, TIME_STEP, 1) para a LSTM
    X_input = torch.from_numpy(scaled_input_data).float().unsqueeze(0)
    
    # Previsão e Desnormalização (Lógica PyTorch)
    with torch.no_grad():
        model.eval() 
        scaled_prediction_tensor = model(X_input)
        
    scaled_prediction = scaled_prediction_tensor.cpu().numpy()
    prediction_original = scaler.inverse_transform(scaled_prediction)[0, 0]

    # 4. Cálculo e Log
    ultimo_preco = recent_prices[-1]
    variacao_pct = ((prediction_original - ultimo_preco) / ultimo_preco) * 100

    # Loga o sucesso no módulo de estado
    state.PREDICTION_COUNTER.inc() 
    
    elapsed = time.time() - start_time
    logger.info(f"Previsão concluída em {elapsed:.3f}s. Preço previsto: {prediction_original:.2f}")

    return PredictionResponse(
        ticker=TICKER,
        ultima_data=ultima_data,
        ultimo_preco=round(float(ultimo_preco), 2),
        previsao_proximo_dia=round(float(prediction_original), 2),
        variacao_percentual=round(float(variacao_pct), 2),
        unidade="R$"
    )