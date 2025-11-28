import os
import joblib
import contextlib 
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime

# Importações dos artefatos
from app.config.settings import SCALER_PATH, MODEL_DIR
from app.model.lstm_light_module import LSTMLightModule
from app.router import prediction_router
from app.config import security
# Importa o módulo de estado que contém as variáveis globais e métricas
from app import state 

# --- 1. FUNÇÃO LIFESPAN: Carregamento e Desligamento de Recursos ---
@contextlib.asynccontextmanager
async def lifespan_startup_shutdown(app: FastAPI):
    """
    Função Lifespan (Substitui on_event). Carrega o modelo e o scaler na inicialização
    e gerencia o desligamento (shutdown), preenchendo as variáveis no módulo state.
    """
    print("Iniciando carregamento do Scaler e Modelo PyTorch...")

    # --- LÓGICA DE STARTUP ---
    # Carregar Scaler
    try:
        # Atribui o objeto carregado ao módulo state
        state.SCALER = joblib.load(SCALER_PATH)
        print(f"Scaler carregado com sucesso de: {SCALER_PATH}")
    except Exception as e:
        print(f"ERRO: Falha ao carregar o Scaler de {SCALER_PATH}. {e}")
        raise RuntimeError("API não pode iniciar sem o scaler.")

    # Encontrar e Carregar o Modelo PyTorch Lightning (.ckpt)
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".ckpt")]
        if not model_files:
             raise FileNotFoundError(f"Nenhum arquivo .ckpt encontrado em {MODEL_DIR}")
        
        # Pega o melhor/último arquivo modificado
        state.BEST_MODEL_PATH = os.path.join(MODEL_DIR, sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)[0])

        # Carrega o LSTMLightModule (hparams usados para reconstruir a arquitetura)
        hparams = {'input_size': 1, 'hidden_size': 50, 'dropout_rate': 0.2, 'learning_rate': 0.001}
        # Atribui o objeto ao módulo state
        state.MODEL = LSTMLightModule.load_from_checkpoint(state.BEST_MODEL_PATH, hparams=hparams)
        state.MODEL.eval() # Coloca o modelo em modo de avaliação
        
        print(f"Modelo PyTorch carregado com sucesso de: {state.BEST_MODEL_PATH}")
        print("API pronta para receber requisições.")
        
    except Exception as e:
        print(f"ERRO: Falha ao carregar o Modelo de {MODEL_DIR}. {e}")
        raise RuntimeError("API não pode iniciar sem o modelo.")
        
    # O 'yield' sinaliza que o startup terminou e a API está pronta para servir
    yield
    
    # --- LÓGICA DE SHUTDOWN (Executado quando o servidor desliga) ---
    print("Desligando e limpando recursos da API...")
    # Limpa as referências no módulo state
    state.MODEL = None
    state.SCALER = None


# --- 2. Inicialização do FastAPI (Passando o Lifespan) ---
app = FastAPI(
    title="API PETR4",
    version="1.0.0",
    description="API para previsão de preços de fechamento da PETR4 usando modelo LSTM.",
    lifespan=lifespan_startup_shutdown
)


# --- REQUISITO 5.1: Middleware de Monitoramento ---
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """
    Middleware que rastreia o tempo e o status de cada requisição (Requisito 5).
    """
    start_time = datetime.now()
    endpoint = request.url.path
    method = request.method
    
    response = await call_next(request)
    
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds()
    
    if endpoint != '/metrics':
        # Usa as métricas do módulo state
        state.REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
    
    # Usa as métricas do módulo state
    state.REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=response.status_code).inc()
    
    return response


# --- Rota de Boas Vindas/Verificação ---
@app.get("/", tags=["Health"])
def home():
    """Health endpoint"""
    # Exibe o caminho do modelo carregado (informação de saúde/versão)
    return {"message": "API de Previsão PETR4 está Online.", "model_path": state.BEST_MODEL_PATH.split('\\')[-1]}

# Endpoint para métricas Prometheus
@app.get("/metrics", tags=["Monitoramento"])
def metrics():
    """Endpoint que expõe as métricas coletadas (Prometheus)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Rota de Previsão
app.include_router(prediction_router.router, tags=["Previsão"])

# Autenticação
app.include_router(security.router)