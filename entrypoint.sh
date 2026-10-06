#!/bin/bash

# Start SSH service
service ssh start

# Start Ollama service in background
ollama serve &

echo "Container started successfully!"
echo "SSH is running on port 22 (mapped to host 2222). Password: root1234"
echo "Ollama is running on port 11434."

# Start FastAPI server (uncomment for production)
# uvicorn main:app --host 0.0.0.0 --port 8000 &

# Keep container running indefinitely
tail -f /dev/null
