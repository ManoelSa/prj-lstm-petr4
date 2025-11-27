import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def create_sequences(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Estrutura uma série temporal unidimensional em sequências de entrada (X) e saídas (Y).
    
    Args:
        data (np.ndarray): Array 1D ou 2D (N, 1) dos dados escalonados.
        time_step (int): O número de passos anteriores a serem usados para a previsão.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, Y) arrays prontos para o treino.
        X shape: (num_amostras, time_step, 1)
        Y shape: (num_amostras, 1)
    """
    X, Y = [], []
    # Converte para 1D se for 2D (N, 1)
    if data.ndim > 1 and data.shape[1] == 1:
        data = data.flatten()

    for i in range(len(data) - time_step):
        # Pega a janela de entrada
        sequence = data[i:(i + time_step)]
        # O alvo é o próximo passo
        target = data[i + time_step]
        X.append(sequence)
        Y.append(target)
        
    X = np.array(X)
    Y = np.array(Y)
    
    # Adiciona a dimensão da feature (1) no final para o PyTorch/LSTM (batch, seq, feature)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
    # Y também deve ter 2D para ser um tensor (batch, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
        
    return X, Y

class TimeSeriesDataset(Dataset):
    """
    A classe Dataset do PyTorch para lidar com dados de séries temporais.
    Converte arrays Numpy em tensores PyTorch.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Inicializa o Dataset.

        Args:
            X (np.ndarray): Array de entradas (features) Numpy.
            Y (np.ndarray): Array de saídas (alvos) Numpy.
        """
        # Converte para float32, necessário para treinamento de DL
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        """
        Retorna o número total de amostras no dataset.

        Returns:
            int: Número de amostras.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna um par (entrada, alvo) para um dado índice.

        Args:
            idx (int): Índice da amostra.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: O par (X[idx], Y[idx]).
        """
        return self.X[idx], self.Y[idx]