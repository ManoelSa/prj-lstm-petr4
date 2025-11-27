import os
from datetime import datetime, timedelta

# --- Configurações de Dados e Treinamento ---
TICKER = "PETR4.SA"
START_DATE = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d') #dinâmica (3 anos atrás)
TIME_STEP = 60 
TEST_SIZE_RATIO = 0.20
EPOCHS = 50 
BATCH_SIZE = 64

# --- Hiperparâmetros do LSTM (para o LSTMLightModule) ---
HIDDEN_SIZE = 50 
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001 # Taxa de aprendizado Adam


# --- Caminhos dos Arquivos (Nova Estrutura 'artifacts') ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diretório base para todos os artefatos gerados (modelos e scalers)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Caminho para os Checkpoints do PyTorch Lightning
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints") 
os.makedirs(MODEL_DIR, exist_ok=True)

# Caminho para o Scaler (essencial para o deploy)
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")


# --- Configuração MLflow---
# Local onde os experimentos serão rastreados. Padrão: local 'mlruns' dentro de ARTIFACTS_DIR
#MLFLOW_TRACKING_URI = os.path.join(ARTIFACTS_DIR, "mlruns")
MLFLOW_DB_PATH = os.path.join(ARTIFACTS_DIR, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

MLFLOW_ARTIFACTS_STORE = os.path.join(ARTIFACTS_DIR, "mlruns")