from prometheus_client import Counter, Histogram

# --- VARIÁVEIS GLOBAIS DE ESTADO (MODELO E SCALER) ---
# Este módulo armazena o estado global da aplicação
MODEL = None
SCALER = None
BEST_MODEL_PATH = "Aguardando Carregamento..."

# --- VARIÁVEIS GLOBAIS DE MONITORAMENTO (PROMETHEUS) ---
# Métricas também centralizadas aqui

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'Latência das requisições HTTP', 
    ['method', 'endpoint']
)
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Contagem total de requisições HTTP', 
    ['method', 'endpoint', 'http_status']
)
PREDICTION_COUNTER = Counter(
    'prediction_count', 
    'Contagem de previsões bem-sucedidas'
)