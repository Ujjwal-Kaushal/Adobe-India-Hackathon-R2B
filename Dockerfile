# Dockerfile

# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for performance and CPU-only builds
ENV TOKENIZERS_PARALLELISM=false
# This forces llama-cpp-python to build without GPU/hardware acceleration hooks
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DLLAMA_HIPBLAS=OFF -DLLAMA_BLAS=OFF"
ENV FORCE_CMAKE=1

# Install system dependencies that might be required for building python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
# The --extra-index-url from the file will be used here
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Run the model initializer script during the image build process
RUN python initialize_model.py

# Set the entrypoint to run the main challenge script.
ENTRYPOINT ["python", "-u", "run_challenge.py"]