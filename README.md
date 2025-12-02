# 📈 Previsão de Séries Temporais Financeiras (PETR4.SA)

## 🎯 Objetivo

Este projeto atende ao desafio de desenvolver um modelo preditivo baseado em **LSTM (Long Short-Term Memory)** para prever o **preço de fechamento (D+1)** de uma ação da bolsa — aqui, **PETR4.SA**.

A solução implementa **toda a pipeline completa**, abrangendo:

- coleta e preparação de dados,
- modelagem e treinamento,
- rastreamento de experimentos,
- deploy do modelo em uma API REST para inferência,
- monitoramento contínuo do serviço.

O projeto demonstra:

- **Modelo LSTM (PyTorch + PyTorch Lightning)** integrado a um pipeline de treinamento estruturado.
- **Rastreabilidade com MLflow:** parâmetros, métricas, artefatos.
- **Deploy escalável** via FastAPI com monitoramento por Prometheus.

---

## 🌟 Arquitetura da Solução (MLOps)

A arquitetura segue o princípio de **serviços desacoplados**, separando **treinamento** de **inferência** para garantir escalabilidade e governança.

### Componentes

- **Treinamento — `train.py` + PyTorch Lightning**  
  Responsável por coleta de dados, criação do dataset, construção e treinamento do modelo LSTM.

- **Inferência — FastAPI (`app/api/main.py`)**  
  API REST que serve o modelo em produção com baixa latência.

- **Rastreabilidade — MLflow (SQLite)**  
  Registra cada execução com métricas, hiperparâmetros e artefatos.

- **Monitoramento — Prometheus**  
  Coleta métricas como latência das requisições e MAE do modelo.

---

## 🛠️ Detalhes de Implementação

### Linguagem e Frameworks

- **Python 3.12.6**
- **PyTorch** – implementação da rede LSTM  
- **PyTorch Lightning** – estrutura e treinamento modularizado  
- **FastAPI** – API REST de inferência  
- **MLflow** – tracking de experimentos  

### Principais dependências

- `pandas`, `numpy`, `datetime` – manipulação de dados  
- `yfinance` – coleta de dados financeiros  
- `scikit-learn` – `MinMaxScaler`  
- `torchmetrics` – métricas (MAE)  
- `prometheus_client` – monitoramento de serviço  
- `joblib` – carregamento/salvamento de scaler 
* `uvicorn` (Servidor ASGI). 

### Arquitetura do Modelo

- **LSTM empilhada (duas camadas)** para previsão de séries temporais  
- **`LSTMFactory`** – módulo com a arquitetura da rede  
- **`LSTMLightModule`** – módulo Lightning que gerencia ciclo de treino, validação e teste  

---

## ✨ Principais Conceitos Técnicos

### 🧠 Modelo LSTM

- Arquitetura de duas camadas LSTM com integração ao PyTorch Lightning.
- Reprodutibilidade assegurada com `pl.seed_everything(42)` e `shuffle=False` no DataLoader.

### 💾 Deploy (FastAPI)

- Treinamento via CLI e inferência online desacoplada.
- Carregamento único do modelo e scaler no `lifespan`, garantindo baixa latência.
- `state.py` mantém instâncias globais de `MODEL` e `SCALER` acessíveis a toda a API.

---

## 📈 Monitoramento e Evolução (Sustentabilidade)

### 🔁 Retreinamento (MLOps)

- **MAE em escala original** é a métrica crítica para avaliar a saúde do modelo.
- A arquitetura já possui:
  - **medição** (Prometheus)  
  - **ação** (`train.py`)  
- Futuro: automatizar o retreinamento acionado por alertas de MAE.

### ⏱️ Monitoramento de SLA

- **GET `/metrics`** expõe métricas de latência e contagem de requisições.
- O SLA pode ser acompanhado via Prometheus/Grafana.
- O modelo está pronto para futura **quantização** (redução de latência).

---

## 🗂️ Estrutura do Projeto

```text
prj-lstm-petr4/
├── app/
│   ├── api/          
│   │   ├── router/
│   │   ├── main.py      # FastAPI com Lifespan e Middleware
│   │   └── state.py     # Estado global (MODEL/SCALER)
│   ├── artifacts/        # Checkpoints, scaler.pkl, mlflow.db
│   ├── config/           # Settings
│   ├── data/             # Pipeline de dados (coleta, Dataset)
│   ├── model/            # LSTMFactory e LSTMLightModule
│   ├── schemas/          # Entradas/Saídas da API (Pydantic)
│   └── utils/            # Funções auxiliares
├── mlruns/               # Artefatos MLflow
├── requirements.txt
└── train.py              # Script de Treinamento
```
## 💻 Execução Local (Windows + Docker + API + Monitoramento)
Esta aplicação faz uso de Prometheus e Grafana para monitoramento em tempo real.
No Windows, é necessário utilizar o Docker Desktop para subir os contêineres automaticamente.
Você pode rodar tudo manualmente ou simplesmente utilizar o script **start_ambiente.py**, que:
- Inicia o Docker Desktop
  - Importante já constar instalado, Baixe em: https://www.docker.com/products/docker-desktop/ 
- Sobe (ou cria, se não existir) os contêineres **prometheus_petr4** e **grafana_petr4**
- Abre automaticamente as URLs no navegador
  - Host Grafana: http://localhost:3000
  - Host Prometheus: http://localhost:9090/targets
- Inicia o servidor FastAPI (Uvicorn)
  - Host API: http://127.0.0.1:8000/docs
### 🚀 Passos para Execução Local

```bash
# 1. Clone o repositório
git clone https://github.com/ManoelSa/prj-lstm-petr4.git
cd prj-lstm-petr4

# 2. (Opcional) Crie e ative um ambiente virtual
python -m venv venv
venv\Scripts\activate #Linux: source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o pipeline de treinamento
python train.py
# Saída esperada: O modelo será treinado por 50 épocas e os artefatos serão salvos em 'artifacts/'

# 6. Inicie o servidor Uvicorn (usando o módulo 'api.main')
uvicorn api.main:app --reload
# Saída esperada: O servidor irá iniciar e carregar o modelo PyTorch com sucesso.


```
