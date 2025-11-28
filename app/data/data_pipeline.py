import yfinance as yf
from datetime import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler
from app.data.dataset import create_sequences, TimeSeriesDataset 
from app.config.settings import TICKER, START_DATE, TIME_STEP, TEST_SIZE_RATIO, SCALER_PATH, MODEL_DIR, BATCH_SIZE 
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DataPipeline(pl.LightningDataModule):
    """
    DataPipeline (LightningDataModule) para PyTorch Lightning.
    Gerencia o download, pré-processamento, divisão e fornecimento de DataLoaders.
    """
    def __init__(self, time_step: int, test_size_ratio: float, batch_size: int, scaler_path: str):
        """
        Inicializa o DataPipeline com hiperparâmetros de dados.

        Args:
            time_step (int): O tamanho da janela de tempo para as sequências.
            test_size_ratio (float): Proporção dos dados a ser usada para o conjunto de teste.
            batch_size (int): O tamanho do lote de dados para o DataLoader.
            scaler_path (str): Caminho onde o MinMaxScaler será salvo.
        """
        super().__init__()
        self.time_step = time_step
        self.test_size_ratio = test_size_ratio
        self.batch_size = batch_size
        self.scaler_path = scaler_path
        
        # Flags de estado para evitar duplicação de I/O pesado
        self.data_prepared = False 

        # Variáveis de dados
        self.X_treino, self.X_teste, self.Y_treino, self.Y_teste = None, None, None, None
        self.X_val, self.Y_val = None, None
        self.scaler = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        """
        Prepara dados: Coleta, escala, salva o scaler e divide os dados.
        Contém uma lógica para garantir que a coleta de dados (I/O pesado) ocorra apenas uma vez.
        """
        # --- LÓGICA DE FLAG: Evita a coleta de dados duplicada ---
        if self.data_prepared:
            return

        print(f"--- 1. Coletando dados para {TICKER} ---")
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            dados_originais = yf.download(TICKER, start=START_DATE, end=end_date, auto_adjust=True)
            if dados_originais.empty:
                raise ValueError("Dataset vazio. Verifique o ticker.")
        except Exception as e:
            print(f"Erro na coleta de dados: {e}")
            return

        # 1. Seleção da feature e Escalamento
        dados_fechamento = dados_originais[['Close']].copy()
        # Scaler: normaliza os dados para a faixa [0, 1] e melhora o desempenho do modelo
        self.scaler = MinMaxScaler(feature_range=(0, 1)) 
        dados_escalonados = self.scaler.fit_transform(dados_fechamento['Close'].values.reshape(-1, 1))

        # 2. Estruturação em Sequências (X e Y)
        X, Y = create_sequences(dados_escalonados, self.time_step)

        # 3. Divisão Treino/Teste/Validação
        train_val_size = int(len(X) * (1 - self.test_size_ratio))
        val_split = 0.2
        train_size = int(train_val_size * (1 - val_split))
        
        self.X_treino, self.X_val = X[0:train_size], X[train_size:train_val_size]
        self.Y_treino, self.Y_val = Y[0:train_size], Y[train_size:train_val_size]
        self.X_teste, self.Y_teste = X[train_val_size:len(X)], Y[train_val_size:len(Y)]

        # 4. Salvamento do Scaler (Requisito 3)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Scaler salvo em: {self.scaler_path}")
        
        # Marca que o I/O pesado foi concluído
        self.data_prepared = True

    def setup(self, stage: str):
        """
        Cria os objetos Dataset a partir dos dados pré-processados (sem I/O pesado).
        """
        # Chama prepare_data para garantir que os dados estejam carregados (se não estiverem)
        self.prepare_data() 
            
        if stage == 'fit':
            self.train_dataset = TimeSeriesDataset(self.X_treino, self.Y_treino)
            self.val_dataset = TimeSeriesDataset(self.X_val, self.Y_val)
        
        if stage == 'test':
            self.test_dataset = TimeSeriesDataset(self.X_teste, self.Y_teste)

    def train_dataloader(self) -> DataLoader:
        """
        Retorna o DataLoader para o conjunto de treinamento.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        """
        Retorna o DataLoader para o conjunto de validação.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """
        Retorna o DataLoader para o conjunto de teste.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_scaler(self) -> MinMaxScaler:
        """
        Retorna a instância do MinMaxScaler para desnormalização.
        """
        return self.scaler