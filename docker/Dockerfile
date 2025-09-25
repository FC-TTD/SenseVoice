# Use the official PyTorch image with CUDA support
#FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
# 设置环境变量，避免生成缓存文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

  # Install runtime utilities for healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#fix bug

  # Install the dependencies
RUN pip install --no-cache-dir onnx onnxconverter_common \
      transformers==4.49.0 pyloudnorm scipy torch==2.3 \
      torchaudio==2.3.0 modelscope huggingface huggingface_hub funasr \
      numpy==1.26.4 gradio

# Install private health plugin from index (package provided by internal PyPI)
RUN pip install --no-cache-dir \
    --extra-index-url http://pypi-server/simple/ --trusted-host pypi-server \
    ttd_fastapi_utils

# Copy the application code into the container
COPY . .

  # Expose the port FastAPI will run on
  EXPOSE 8000
  
# Define environment variables（采用 uvicorn 多 worker + Torch 自动线程）
ENV TMP_DIR=/app/tmp
ENV UVICORN_TIMEOUT=600
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120
ENV UVICORN_WORKERS=4

# Create the temporary directory
RUN mkdir -p $TMP_DIR

VOLUME /root/.cache
HEALTHCHECK --interval=30s --timeout=5s --start-period=1m --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# 使用 uvicorn 多 worker 启动，workers 等参数从环境变量控制
CMD sh -lc 'uvicorn api:app --host 0.0.0.0 --port 8000 --log-level info --workers ${UVICORN_WORKERS} --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE}'
