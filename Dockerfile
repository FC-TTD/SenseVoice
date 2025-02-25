# Use the official Python 3.12 base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi[standard] onnx onnxconverter_common opensmile

# Copy the rest of the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Define environment variables
ENV TMP_DIR=/app/tmp

# Create the temporary directory
RUN mkdir -p $TMP_DIR

# Set the command to run the FastAPI server
CMD ["fastapi", "run", "--port", "8000"]