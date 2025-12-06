# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements files first (for better caching)
COPY requirements.txt .
COPY CoMEM-Agent-Inference/requirements_web.txt ./CoMEM-Agent-Inference/

# Install PyTorch first (required by other packages)
RUN pip install --no-cache-dir torch==2.7.1

# Install vllm first (it will pull in compatible transformers version)
RUN pip install --no-cache-dir vllm==0.11.0

# Install other ML/AI packages
RUN pip install --no-cache-dir peft wandb qwen_vl_utils qwen_agent openai anthropic

# Install basic Python dependencies
RUN pip install --no-cache-dir beartype playwright Pillow pandas gymnasium Matplotlib dashscope bs4 \
    python-dotenv faiss-cpu evaluate nltk crawl4ai accelerate

# Install web/networking libraries
RUN pip install --no-cache-dir aiofiles aiohappyeyeballs aiohttp aiolimiter aiosignal aiosqlite \
    Brotli certifi chardet charset-normalizer httpcore httpx requests urllib3 websocket-client

# Install data processing libraries
RUN pip install --no-cache-dir datasets pyarrow protobuf

# Install other utilities
RUN pip install --no-cache-dir patchright tf-playwright-stealth beautifulsoup4 lxml cssselect \
    click click-log GitPython rich pydantic tiktoken tqdm humanize sentry-sdk

# Note: flash_attn is skipped as it requires complex compilation and is often optional for inference
# Note: Some packages from requirements_web.txt are already installed via dependencies

# Install Playwright and its dependencies
RUN playwright install --with-deps chromium

# Copy the rest of the project
COPY . .

# Set the default command
CMD ["/bin/bash"]
