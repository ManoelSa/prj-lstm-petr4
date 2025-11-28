from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """
    Schema de saída da previsão de preço da ação.
    """
    ticker: str = Field(..., description="Ticker da ação (e.g., PETR4.SA).")
    ultima_data: str = Field(..., description="Data do último preço de fechamento usado para a previsão.")
    ultimo_preco: float = Field(..., description="Último preço de fechamento (R$).")
    previsao_proximo_dia: float = Field(..., description="Preço previsto para o próximo dia útil (R$).")
    variacao_percentual: float = Field(..., description="Variação percentual entre o último preço e a previsão.")
    unidade: str = Field("R$", description="Unidade monetária.")