# Passo 01: Escolher a imagem base
FROM python:3.11-slim 
# -slim: traz uma versão mais leve do python

# Passo 02: Definir o diretorio de trabalho do container
WORKDIR /app

# Passo 03: instalação de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: evita que o pip armazene os arquivos de instalacao no cache


# Passo 04: Copiar o codigo para o container
COPY src/ src
COPY data/ data/
# Estrutura do projeto dentro do container:
# /app
# -src (todo o projeto)
# -data

# Passo 05: definir variaveis de ambiente
ENV DB_PATH="heat_exchanger.db"\
    MODEL_PATH="artifacts/heat_efficiency_model.pkl"

# Para realizar inferencias, podemos bypassar o comando de execucao indicado
# docker run docker-ml-projeto sh -c "python3 src/train.py && python3 src/inference.py --efficiency 300" 

# Passo 06: definir o comando de execucao dentro do container
CMD ["python", "src/train.py"]
