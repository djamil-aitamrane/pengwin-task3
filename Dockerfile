# PENGWIN 2026 Task 3 — inference container
FROM python:3.10-slim

# 1. libs systeme (root)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# 2. deps python (root) — pip a jour D'ABORD (corrige le wheel typing_extensions)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir trimesh==4.8.2 "numpy>=1.24,<2.3" "scipy>=1.10" "PyYAML>=6.0"

# 3. code (root) — model_test.py inclus
COPY data_io.py preprocess.py model.py model_test.py inference.py ./

# 4. user non-root + points de montage + ownership (root)
RUN groupadd -r user && useradd -r -g user user && \
    mkdir -p /input /output /opt/ml/model && \
    chown -R user:user /opt/app /input /output /opt/ml/model

# 5. bascule non-root a la fin
USER user

ENTRYPOINT ["python", "inference.py", "--input_dir", "/input", \
            "--output", "/output/reduction-poses-matrices.json", \
            "--model", "/opt/ml/model"]
