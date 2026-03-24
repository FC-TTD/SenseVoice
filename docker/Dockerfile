# Use the official PyTorch image with CUDA support
#FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
# 设置环境变量，避免生成缓存文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

  # Install runtime utilities for healthcheck
RUN apt-get update -o Acquire::Retries=5 -o Acquire::ForceIPv4=true \
    && apt-get install -y --fix-missing --no-install-recommends curl ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#fix bug

  # Install the dependencies in a single uv virtualenv
RUN pip install --no-cache-dir uv \
    && uv venv ${VIRTUAL_ENV} \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python --no-cache \
      --default-index https://pypi.org/simple \
      --index https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match \
      onnx onnxconverter_common transformers==4.49.0 pyloudnorm scipy \
      torch==2.8.0+cpu torchaudio==2.8.0+cpu torchcodec==0.7.0 \
      modelscope huggingface huggingface_hub funasr whisperx==3.8.2 \
      gradio==5.50.0 "fastapi>=0.111.1" uvicorn requests \
      soundfile librosa \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python --no-cache \
      --extra-index-url http://pypi-server/simple/ --trusted-host pypi-server \
      "ttd_fastapi_utils>=0.2.1"

# Copy the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000
  
# Define environment variables（单进程 + 动态线程池 + 延迟加载模型）
ENV TMP_DIR=/app/tmp
ENV UVICORN_TIMEOUT=600
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120
ENV UVICORN_WORKERS=1
ENV MAX_CONCURRENT_REQUESTS=32
ENV MODEL_IDLE_TIMEOUT=1800
ENV ENABLE_AUTO_UNLOAD=true
ENV TTD_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4e5f3845-cd5f-43a0-b004-231e710c68e4"
ENV SENSEVOICE_WHISPERX_MODEL=small
ENV SENSEVOICE_WHISPERX_VAD_METHOD=silero
ENV SENSEVOICE_WHISPERX_ALIGN=true
ENV SENSEVOICE_WHISPERX_BATCH_SIZE=8
ENV SENSEVOICE_WHISPERX_THREADS=8

# Create the temporary directory
RUN mkdir -p $TMP_DIR

VOLUME /root/.cache

# 使用 uvicorn 多 worker 启动，workers 等参数从环境变量控制
CMD ["sh", "-c", "exec /opt/venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --log-level info --workers ${UVICORN_WORKERS} --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE}"]
