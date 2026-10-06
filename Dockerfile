FROM python:3.12.4-slim

# Install system utilities, SSH, curl, git, zstd, wget, util-linux (flock), tar, gzip
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    openssh-server \
    git \
    procps \
    zstd \
    wget \
    util-linux \
    tar \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama inside container
RUN curl -fsSL https://ollama.com/install.sh | sh

# Configure SSH server
RUN mkdir /var/run/sshd
RUN echo 'root:root1234' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all python code into the container
COPY . /app

# Expose SSH and Ollama ports
EXPOSE 22 11434

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Pre-download HuggingFace tokenizer for true Air-Gap execution
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-14B')"

CMD ["/entrypoint.sh"]
