import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

APP_USER = os.getenv("APP_USER")
APP_PASS = os.getenv("APP_PASS")

LOGIN_URL = "http://127.0.0.1:8000/login"
PREDICT_URL = "http://127.0.0.1:8000/predict/petr4"

data = {
    "username": APP_USER,
    "password": APP_PASS
}

def get_token():
    """Realiza login e retorna o token JWT."""
    try:
        response = requests.post(url=LOGIN_URL, data=data)
        response.raise_for_status()

        token = response.json().get("access_token")
        print(f"Token obtido: {token[:20]}...")
        return token
    except Exception as e:
        print("Erro ao fazer login:", e)
        return None

def call_predict(token):
    """Chama a rota de previsão usando o token JWT."""
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.post(PREDICT_URL, headers=headers)
        print("Status:", response.status_code)
        print("Resposta:", response.json())
        print("-" * 50)

    except Exception as e:
        print("Erro ao chamar /predict:", e)


if __name__ == "__main__":
    print("Iniciando chamadas a cada 5 segundos...")

    token = get_token()


    if not token:
        print("Não foi possível obter token. Encerrando.")
        exit()

    while True:
        call_predict(token)
        time.sleep(5)