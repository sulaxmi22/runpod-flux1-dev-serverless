#!/usr/bin/env bash
# Build the Docker image and push it to Docker Hub / Runpod Template workflow.
set -e

IMAGE_NAME="${IMAGE_NAME:-flux1-dev-runpod-serverless}"
DOCKER_USERNAME="${DOCKER_USERNAME:-your-dockerhub-username}"
TAG="${TAG:-latest}"
FULL_TAG="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_TAG}"
docker build -t "${FULL_TAG}" .

echo "Pushing image to registry..."
docker push "${FULL_TAG}"

echo "Image published: ${FULL_TAG}"
echo "Use this in your Runpod Serverless endpoint template."
