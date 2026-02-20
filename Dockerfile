# Usar imagen base de Python
FROM python:3.11.14-bookworm

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .

# Instalar uv (el instalador oficial)
RUN apt-get update && apt-get install -y \
    build-essential \
    # python3-dev \
    # libpq-dev \
    cmake \
    # git \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Copiar el proyecto completo
COPY . .

# Exposer puerto si es necesario (para uvicorn/API)
# EXPOSE 8000

# Comando por defecto
CMD ["python", "main.py"]
