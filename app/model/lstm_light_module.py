import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from app.model.lstm_factory import LSTMFactory 
from torchmetrics.regression import MeanAbsoluteError

class LSTMLightModule(pl.LightningModule):
    """
    Módulo de treinamento PyTorch Lightning. 
    Encapsula o modelo (via LSTMFactory), a função de perda, o otimizador e a lógica de treinamento.
    """
    def __init__(self, hparams: dict):
        """
        Inicializa o módulo e configura o modelo e a função de perda.

        Args:
            hparams (dict): Dicionário de hiperparâmetros, incluindo 'input_size', 'hidden_size', 
                            'dropout_rate', e 'learning_rate'.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # 1. Criação do Modelo (via Factory)
        self.model = LSTMFactory(
            input_size=hparams.get('input_size', 1),
            hidden_size=hparams.get('hidden_size', 50),
            dropout_rate=hparams.get('dropout_rate', 0.2)
        )
        
        # 2. Função de Perda: MSE (Mean Squared Error) e Métrica MAE
        self.loss_fn = nn.MSELoss()
        self.mae_metric = MeanAbsoluteError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define a passagem adiante do módulo.

        Args:
            x (torch.Tensor): Tensor de entrada no formato (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Previsões de saída no formato (batch_size, 1).
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configura e retorna o otimizador a ser usado no treinamento.

        Returns:
            optim.Optimizer: Otimizador Adam.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.get('learning_rate', 1e-3)) 
        return optimizer
        
    def training_step(self, batch, batch_idx):
        """
        Define a lógica para um único passo de treinamento.

        Args:
            batch (tuple): Lote de dados (X, Y) do DataLoader.
            batch_idx (int): Índice do lote.

        Returns:
            torch.Tensor: O valor da função de perda de treinamento.
        """
        return self._common_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        """
        Define a lógica para um único passo de validação.

        Args:
            batch (tuple): Lote de dados (X, Y) do DataLoader.
            batch_idx (int): Índice do lote.

        Returns:
            torch.Tensor: O valor da função de perda de validação.
        """
        return self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        """
        Define a lógica para um único passo de teste/avaliação final.

        Args:
            batch (tuple): Lote de dados (X, Y) do DataLoader.
            batch_idx (int): Índice do lote.

        Returns:
            dict: Dicionário contendo a perda, MAE, previsões e valores verdadeiros.
        """
        loss = self._common_step(batch, 'test')
        
        # Re-calcula y_pred para o dicionário de retorno (necessário para desnormalização)
        x, y = batch
        y_pred = self(x)
        
        # Loga RMSE (raiz quadrada do MSE)
        self.log('test_rmse', loss.sqrt(), on_step=False, on_epoch=True) 
        
        return {"loss": loss, "mae": self.mae_metric.compute(), "y_pred": y_pred, "y_true": y.float().view(-1, 1)}

    # Função auxiliar
    def _common_step(self, batch: tuple, stage: str):
        """
        Lógica comum de cálculo de perda e métricas para treino e validação.

        Args:
            batch (tuple): Um par (x, y) de tensores do DataLoader.
            stage (str): O estágio atual ('train', 'val' ou 'test').
            
        Returns:
            torch.Tensor: O valor da função de perda (Loss).
        """
        x, y = batch
        y_true = y.float().view(-1, 1) 
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        mae = self.mae_metric(y_pred, y_true)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True)
        self.log(f'{stage}_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss