# Use the official PyTorch image with CUDA support
#FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# 设置环境变量，避免生成缓存文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#fix bug
#RUN rm -rf /opt/conda/lib/python3.10/site-packages/setuptools/__pycache__ \
#    && python -m pip install --upgrade setuptools

# Install the dependencies
RUN pip install --no-cache-dir onnx onnxconverter_common \
    transformers==4.49.0 pyloudnorm scipy torch==2.3 \
    torchaudio==2.3.0 modelscope huggingface huggingface_hub funasr==1.1.3 \
    numpy==1.26.4 fastapi[standard]

# Copy the application code into the container
# 注意：这里我们不再直接复制所有文件，而是在CI/CD过程中通过rsync控制复制内容
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Define environment variables
ENV TMP_DIR=/app/tmp
ENV UVICORN_TIMEOUT=600
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120

# Create the temporary directory
RUN mkdir -p $TMP_DIR

VOLUME /root/.cache

# 使用 uvicorn 直接启动应用，确保日志正确输出
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
