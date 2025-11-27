import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger 
import mlflow 
from dotenv import load_dotenv

# Importação dos novos módulos PyTorch/Lightning
from app.model.lstm_light_module import LSTMLightModule
from app.data.data_pipeline import DataPipeline 
from app.utils.helpers import evaluate_predictions 
import joblib 

# Importações de Configurações
from app.config.settings import (
    TIME_STEP, EPOCHS, BATCH_SIZE, SCALER_PATH, 
    LEARNING_RATE, HIDDEN_SIZE, DROPOUT_RATE, 
    TEST_SIZE_RATIO, MODEL_DIR, MLFLOW_TRACKING_URI, 
    MLFLOW_ARTIFACTS_STORE, TICKER, START_DATE
)

load_dotenv()

# --- Configuração de Reproducibilidade ---

def set_global_seed(seed):
    """Fixa as seeds globais para numpy, python, e pytorch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


if __name__ == "__main__":
    print("\n")
    AMBIENTE = os.getenv("PIPELINE_AMBIENTE", "dev")
    SEED = int(os.getenv("SEED", 42))

    if AMBIENTE == "dev":
        set_global_seed(SEED)
        print(f"AMBIENTE DEV: Seed global fixada em {SEED}")
    else:
        print("AMBIENTE PRD: Seed não fixada (execuções variáveis)")

    print("\n")
    print("Iniciando Processo de Treinamento e Avaliação de Modelo LSTM (PyTorch Lightning/MLflow).\n")
    
    # --- 1. Configuração do MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    mlflow_logger = MLFlowLogger(
        experiment_name="LSTM Time Series Forecasting",
        tracking_uri=MLFLOW_TRACKING_URI,
        artifact_location=MLFLOW_ARTIFACTS_STORE,
        log_model=False 
    )
    
    # --- LOG MANUAL DE PARÂMETROS DE DADOS ---
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_param('data_ticker', TICKER)
        mlflow.log_param('data_start_date', START_DATE)
        mlflow.log_param('data_test_ratio', TEST_SIZE_RATIO)
        mlflow.log_param('data_time_step', TIME_STEP)
    # --------------------------------------------------------------------------
    
    # --- 2. Preparação do DataPipeline (Substitui load_and_preprocess_data) ---
    data_pipeline = DataPipeline(
        time_step=TIME_STEP,
        test_size_ratio=TEST_SIZE_RATIO,
        batch_size=BATCH_SIZE,
        scaler_path=SCALER_PATH
    )
    
    # O trainer.fit chamará data_pipeline.prepare_data() automaticamente
    # --- 3. Configuração do Modelo e Hiperparâmetros ---
    hparams = {
        'input_size': 1, 
        'hidden_size': HIDDEN_SIZE, 
        'dropout_rate': DROPOUT_RATE, 
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'time_step': TIME_STEP
    }
    
    modelo_pl = LSTMLightModule(hparams)
    
    # --- 4. Callbacks e Trainer ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss', 
        mode='min',
        save_top_k=1,
    )
    
    trainer = pl.Trainer(
        logger=mlflow_logger,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],
    )

    # --- 5. Treinamento (Substitui modelo.fit) ---
    print(f"\n--- REQ: 2.2 Treinamento ({EPOCHS} Epochs) ---")
    trainer.fit(modelo_pl, datamodule=data_pipeline)
    
    # --- 6. Avaliação (Substitui modelo.evaluate e modelo.predict) ---
    print("\n--- REQ: 2.3 Avaliação e Predição ---")
    test_results = trainer.test(ckpt_path='best', datamodule=data_pipeline)

    # --- 7. Desnormalização e Métricas Finais (Fora do PL) ---
    best_model_path = checkpoint_callback.best_model_path
    
    if best_model_path:
        print(f"\nMelhor modelo carregado de: {best_model_path}")
        modelo_final = LSTMLightModule.load_from_checkpoint(best_model_path, hparams=hparams)
        modelo_final.eval() 
        
        test_dataloader = data_pipeline.test_dataloader()
        all_y_pred = []
        all_y_true = []
        
        with torch.no_grad(): 
            for X_batch, Y_batch in test_dataloader:
                Y_pred_batch = modelo_final(X_batch)
                all_y_pred.append(Y_pred_batch.cpu().numpy())
                all_y_true.append(Y_batch.cpu().numpy())

        Y_previsao = np.concatenate(all_y_pred)
        Y_teste = np.concatenate(all_y_true)

        scaler = joblib.load(SCALER_PATH)
        Y_previsao_original = scaler.inverse_transform(Y_previsao)
        Y_teste_original = scaler.inverse_transform(Y_teste)

        metrics = evaluate_predictions(Y_teste_original, Y_previsao_original)
        
        print("\n--- Métricas Finais Pós-Desnormalização ---")
        print(f"RMSE (Original): {metrics['rmse']:.4f} R$")
        print(f"MAE (Original): {metrics['mae']:.4f} R$")
        print(f"MAPE (Original): {metrics['mape']:.2f} %")
    else:
        print("Nenhum modelo foi treinado/salvo.")
        metrics = {'rmse': test_results[0]['test_rmse'], 'mae': test_results[0]['test_mae']}

    print("\n--- Pipeline de Treinamento Concluída ---")
    print(f"Modelo salvo e métricas registradas no MLflow. MAE final: {metrics['mae']:.4f} R$\n")