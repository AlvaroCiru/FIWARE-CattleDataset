FROM python:3.11

# Establecer el directorio de trabajo
WORKDIR /prediction-job

# Instalar dependencias del sistema necesarias para distutils y compilación de paquetes
RUN apt-get update && apt-get install -y \
    python3-distutils \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY ./requirements.txt ./
COPY cattle_dataset.xlsx /prediction-job/

RUN pip install --upgrade pip wheel && pip install setuptools==65.5.1
RUN pip install -r requirements.txt
RUN pip install --upgrade pip && \
    pip install pandas openpyxl requests
# Copiar el resto del código
COPY . .

# Comando de inicio
CMD ["python", "train.py"]
