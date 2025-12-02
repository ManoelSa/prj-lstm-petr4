import subprocess
import time
import os
import webbrowser

# Excemplo Docker Desktop no Windows
DOCKER_DESKTOP_PATH = r"C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"
TEMPO_DE_ESPERA = 20  # Tempo em segundos para esperar o Docker inicializar

#URLS
GRAFANA_URL = 'http://localhost:3000'
PROMETHEUS_URL = 'http://localhost:9090/targets'

# Comandos
PWD_WINDOWS = os.getcwd().replace('\\', '/')
GRAFANA = ["docker", "start", "grafana_petr4"]
GRAFANA_CONTAINER = ["docker", "run", "-d", "--name", "grafana_petr4", "-p", "3000:3000", "grafana/grafana"]

PROMETHEUS = ["docker", "start", "prometheus_petr4"]
PROMETHEUS_CONTAINER = ["docker", "run", "-d", "--name", "prometheus_petr4","-p", "9090:9090", "-v",f"{PWD_WINDOWS}/prometheus.yml:/etc/prometheus/prometheus.yml","prom/prometheus"]

UVICORN = ["uvicorn", "app.api.main:app", "--reload"]

# Função para Inciar Docker Desktop (Importante ter o app já instalado no ambiente Windows)
def iniciar_docker_desktop():
    """Inicia o Docker Desktop se ele estiver instalado no caminho padrão."""
    if os.path.exists(DOCKER_DESKTOP_PATH):
        print("Iniciando Docker Desktop...")
        try:
            subprocess.Popen(f'"{DOCKER_DESKTOP_PATH}"', shell=True)
            print(f"Aguardando {TEMPO_DE_ESPERA} segundos para o Docker inicializar...")
            time.sleep(TEMPO_DE_ESPERA)
            return True
        except Exception as e:
            print(f"Erro ao iniciar o Docker Desktop: {e}")
            return False
    else:
        print(f"ERRO: Docker Desktop não encontrado em: {DOCKER_DESKTOP_PATH}")
        return False

#Função para Executar Comandos do Docker
def executar_comando_docker(comando, nome_acao):
    """Executa um comando do Docker CLI e verifica o status."""
    print(f"\n--- Tentando: {nome_acao} ---")
    print(f"Executando: {' '.join(comando)}")
    
    # Executa o comando e captura a saída e o código de retorno
    resultado = subprocess.run(comando, capture_output=True, text=True)
    
    if resultado.returncode == 0:
        print(f"SUCESSO: {nome_acao} executado com êxito!")
        print("Saída:", resultado.stdout.strip())
        return True
    else:
        print(f"FALHA: {nome_acao} falhou (Código {resultado.returncode}).")
        print("Erro:", resultado.stderr.strip())
        return False


if __name__ == "__main__":

    if not iniciar_docker_desktop():
        exit(1)
    
    if executar_comando_docker(GRAFANA, "docker start grafana_petr4"):
        print("\n Grafana_petr4 iniciado.")
        print("Abrindo URL do GRAFANA")
        webbrowser.open_new_tab(GRAFANA_URL)     
    else:
        print("\nO contêiner não existe. Criando a imagem e iniciando...")
        if executar_comando_docker(GRAFANA_CONTAINER, "docker run (Criação e Início)"):
            print("\n Grafana_petr4 foi criado e iniciado com sucesso.")
            print("Abrindo URL do GRAFANA")
            webbrowser.open_new_tab(GRAFANA_URL) 
        else:
            print("\n Falha ao criar e iniciar o contêiner Grafana.")
        
    if executar_comando_docker(PROMETHEUS, "docker start prometheus_petr4"):
        print("\n Prometheus_petr4 iniciado.")
        print("Abrindo URL do PROMETHEUS")
        webbrowser.open_new_tab(PROMETHEUS_URL) 
    else:
        print("\nO contêiner não existe. Criando a imagem e iniciando...")
        if executar_comando_docker(PROMETHEUS_CONTAINER, "docker run (Criação e Início)"):
            print("\n Prometheus_petr4 foi criado e iniciado com sucesso.")
            print("Abrindo URL do PROMETHEUS")
            webbrowser.open_new_tab(PROMETHEUS_URL) 
        else:
            print("\n Falha ao criar e iniciar o contêiner Prometheus.")


    #Iniciando Aplicaçao
    print("Iniciando APP")
    subprocess.run(UVICORN, check=True)

    
    
