# NVIDIA NGC PyTorch + TensorRT base. Pin the tag so recipe cards remain reproducible.
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMNIOPTIMIZER_IN_DOCKER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/omnioptimizer

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["bash"]
