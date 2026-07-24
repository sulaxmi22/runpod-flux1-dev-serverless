# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip and install Python dependencies
COPY docker_requirements.txt /app/docker_requirements.txt
RUN pip install --no-cache-dir -r /app/docker_requirements.txt

# Copy handler and any additional project files
COPY handler.py /app/handler.py

# Expose the port Runpod serverless expects (8000 by default)
EXPOSE 8000

# Runpod serverless entry point
CMD ["python", "-u", "/app/handler.py"]
