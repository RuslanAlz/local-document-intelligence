#!/bin/bash

# Start Ollama service in background
ollama serve &

echo "Container started successfully!"
echo "Ollama is running on port 11434."

# Start FastAPI server (uncomment for production)
# uvicorn main:app --host 0.0.0.0 --port 8000 &

# Keep container running indefinitely
tail -f /dev/null
