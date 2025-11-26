# Imagen base oficial de NVIDIA con CUDA 12.1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# 1) Paquetes del sistema + Python + git (para GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git openssh-client \
    build-essential \
    wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) Alias "python" y "pip"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 3) Carpeta de trabajo dentro del contenedor
WORKDIR /workspace

# 4) Copiar dependencias
COPY requirements.txt /workspace/requirements.txt

# 5) Actualizar pip
RUN pip install --upgrade pip

# 6) Instalar PyTorch GPU (CUDA 12.1) compatible con RTX4080 y A100
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 7) Instalar el resto de dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# 8) Crear usuario no root
RUN useradd -ms /bin/bash vscode
USER vscode

WORKDIR /workspace
# --- IGNORE ---