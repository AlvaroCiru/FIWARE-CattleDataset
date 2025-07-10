FROM alpine:3.19.1

# Instalar herramientas del sistema y dependencias de Python
RUN apk --no-cache add \
    python3 \
    py3-pip \
    py3-virtualenv \
    curl \
    bash \
    build-base \
    libffi-dev \
    openssl-dev

# Crear y activar entorno virtual
ENV VENV_PATH=/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copiar los archivos del proyecto
WORKDIR /entities
COPY . .

# Instalar dependencias Python (las que no están por apk)
RUN pip install --upgrade pip && \
    pip install pandas openpyxl requests

# Comando principal
CMD ["sh", "-c", "sh awaitForOrion.sh && sh createPredictionEntities.sh && sh createSubscriptions.sh && python updateCowEntityLoop.py"]
