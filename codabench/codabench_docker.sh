#!/bin/bash
# Script to start 3 Codabench compute workers

# Ensure the .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file containing BROKER_URL not found!"
    echo "Please create a .env file with your BROKER_URL."
    exit 1
fi

echo "Pulling latest Codabench compute worker image..."
docker pull codalab/competitions-v2-compute-worker:cpu1.1

echo "Starting 3 compute workers..."

for i in {1..3}; do
    WORKER_NAME="compute_worker_$i"
    
    # Stop and remove existing container if it exists
    if [ "$(docker ps -aq -f name=$WORKER_NAME)" ]; then
        echo "Stopping and removing existing $WORKER_NAME..."
        docker stop $WORKER_NAME > /dev/null 2>&1
        docker rm $WORKER_NAME > /dev/null 2>&1
    fi
    
    echo "Starting $WORKER_NAME..."
    docker run \
        -v /codabench:/codabench \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -d \
        --env-file .env \
        --name $WORKER_NAME \
        --restart unless-stopped \
        --log-opt max-size=50m \
        --log-opt max-file=3 \
        codalab/competitions-v2-compute-worker:cpu1.1
done

echo "All workers started! You can check logs with: docker logs -f compute_worker_1"
