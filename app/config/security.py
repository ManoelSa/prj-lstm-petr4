import os
import jwt
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi import APIRouter, Depends, HTTPException
from dotenv import load_dotenv
from datetime import datetime, timedelta, UTC
from http import HTTPStatus

load_dotenv()

router = APIRouter()

#Variaveis/Config --  No Futuro trocar por conexao via Banco
APP_USER = os.getenv("APP_USER")
APP_PASS = os.getenv("APP_PASS")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """
    Gera um token JWT assinado com dados fornecidos e tempo de expiração.

    Args:
        data (dict): Dados a serem incluídos no payload do token (ex: {"sub": "usuario"}).
        expires_delta (timedelta | None): Tempo até o token expirar. Se não for informado, o padrão é 15 minutos.

    Returns:
        str: Token JWT codificado como string.
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str = Depends(oauth2_scheme)):
    """
    Valida um token JWT e extrai o nome de usuário (campo 'sub') do payload.

    Args:
        token (str): Token JWT no formato Bearer.

    Returns:
        str: Nome de usuário extraído do token (campo 'sub').

    Raises:
        HTTPException: Se o token estiver expirado, inválido ou não contiver o campo 'sub'.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Token inválido")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Token expirado")
    except jwt.PyJWTError:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Token inválido")
    

@router.post("/login", tags=["Autenticação"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Autentica o usuário com base nas credenciais fornecidas e retorna um token JWT.

    Args:
        form_data (OAuth2PasswordRequestForm): Formulário com `username` e `password` recebido via OAuth2.

    Returns:
        dict: Um dicionário contendo o token de acesso (`access_token`) e o tipo de token (`bearer`).

    Raises:
        HTTPException: Se as credenciais estiverem incorretas.
    """
    user = os.getenv("APP_USER")
    password = os.getenv("APP_PASS")

    if form_data.username != user or form_data.password != password:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Credenciais inválidas")

    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}