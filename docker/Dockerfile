# Imagen base NVIDIA con TensorRT + CUDA + Python
# (elige un tag reciente, por ejemplo 24.08-py3 según lo que tengas disponible en NGC)
FROM nvcr.io/nvidia/tensorrt:24.08-py3

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Asegurar que trtexec esté en el PATH (según docs oficiales)
ENV PATH="/opt/tensorrt/bin:${PATH}"

# 1) Paquetes del sistema + git (para GitHub) + deps de OpenCV/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    git openssh-client \
    build-essential \
    wget curl ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# (En muchos contenedores NGC, 'python' ya apunta al intérprete correcto.
#  Si no fuera así, puedes añadir los update-alternatives como antes.)

# 2) Carpeta de trabajo dentro del contenedor
WORKDIR /workspace

# 3) Copiar dependencias
COPY requirements.txt /workspace/requirements.txt

# 4) Actualizar pip
RUN python -m pip install --upgrade pip

# 5) Instalar PyTorch GPU (CUDA 12.1) si la imagen no lo trae ya
#    Si el contenedor ya trae torch, puedes comentar este bloque.
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 6) Instalar el resto de dependencias del proyecto (incluye onnx y onnxruntime-gpu)
RUN pip install --no-cache-dir -r requirements.txt

# 7) Crear usuario no root (para VSCode devcontainer)
RUN useradd -ms /bin/bash vscode
USER vscode

WORKDIR /workspace
