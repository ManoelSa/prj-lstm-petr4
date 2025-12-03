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

Obs.: Para executar o script **start_ambiente.py** é importante seguir primeiro os passos abaixo.

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
```
## 🔎 Análise de Experimentos com MLflow

Após iniciar os steps atenteriores, é hora de explorar, começando com interface do MLflow:
-  Execute: `mlflow ui --backend-store-uri sqlite:///app/artifacts/mlflow.db`
- Host: `http://127.0.0.1:5000/`

Utilize os seguintes pontos para analisar a performance e a rastreabilidade do modelo:

### 1. Rastreamento e Reprodutibilidade (Parâmetros)

Ao clicar no ID de uma **Run (Execução)**, o primeiro foco é na seção **Parâmetros**.

* **Verificação de Hiperparâmetros:** Confirme que os parâmetros do modelo (`hidden_size`, `dropout_rate`, `learning_rate`) e do treino (`epochs`, `batch_size`) foram logados automaticamente pelo PyTorch Lightning.
* **Verificação de Parâmetros de Dados:** Procure os logs manuais (`data_ticker`, `data_time_step`, `data_start_date`). **Estes comprovam a rastreabilidade:** possibilitando saber exatamente com quais configurações e dados o modelo foi treinado.

### 2. Análise de Desempenho (Métricas)

Utilize a seção **Métricas** para avaliar a qualidade do modelo ao longo do tempo.

* **Curva de `val_loss` (Perda de Validação):** Este é o gráfico mais importante. A curva deve cair de forma consistente e depois se estabilizar. Se a curva começar a subir, indica **overfitting** (o modelo está memorizando o treino e perdendo a capacidade de generalização).
* **Métrica de Produção (`test_mae`):** Verifique o valor final do `test_mae` (Mean Absolute Error). Este valor, que é uma **métrica escalonada**, deve ser baixo. Ele se correlaciona diretamente com o **MAE em R$** calculado na etapa final do `train.py`.

A interface do **MLflow** atua como o Registro de Experimentos (Model Registry), fornecendo um histórico completo para auditoria e garantindo que o modelo seja rastreável e auditável.


## 📊 Monitoramento em Grafana (Análise de Produção)

Para analisar a saúde do serviço (SLA) e a efetividade do modelo, utilizamos o Prometheus e o Grafana, habilitados nos passos anteriores.

Apesar da automação, ainda é necessário realizar duas configurações manuais no Grafana:

1.  **Conexão do Data Source (Prometheus):**
    * Acesse o Grafana (`http://localhost:3000`).
    * Vá para **Data Sources** e adicione o Prometheus.
    * No campo **URL**, utilize o endereço do serviço: `http://host.docker.internal:9090` (Este é o endereço que permite ao Grafana acessar o Prometheus que está rodando no contêiner).
    * Clique em "Save & Test".

2.  **Importação do Dashboard:**
    * Vá para `Dashboards` -> `New` -> `Import`.
    * Selecione o JSON do seu dashboard (localizado na pasta `metrics/`).
    * Na importação, estabeleça um nome e uma pasta para seu dashboard, e fim, pronto para uso.