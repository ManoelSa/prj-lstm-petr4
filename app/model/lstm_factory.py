import torch
import torch.nn as nn

class LSTMFactory(nn.Module):
    """
    Constrói a arquitetura empilhada do modelo LSTM usando módulos do PyTorch.
    """
    def __init__(self, input_size: int, hidden_size: int = 50, dropout_rate: float = 0.2):
        """
        Inicializa as camadas da rede neural.

        Args:
            input_size (int): Número de features de entrada (o 1 no shape (TIME_STEP, 1)).
            hidden_size (int): Número de unidades nas camadas LSTM (50, neste caso).
            dropout_rate (float): Taxa de dropout (0.2, neste caso).
        """
        super().__init__()
        
        # Camada 1: LSTM (50 unidades, return_sequences=True)
        self.lstm1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Camada 2: LSTM (50 unidades, return_sequences=False implícito no forward)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Camada Densa
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define a passagem adiante dos dados (forward pass).

        Args:
            x (torch.Tensor): Tensor de entrada no formato (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Previsões de saída no formato (batch_size, 1).
        """
  
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Seleciona o output do último passo da sequência
        last_output = lstm_out2[:, -1, :] 
        
        last_output = self.dropout2(last_output)
        
        output = self.linear(last_output)
        
        return output