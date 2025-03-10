# Base image: Ubuntu 22.04 (Jammy)
FROM ubuntu:22.04

# Set up timezone (prevent interactive prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update package list and install essential tools
RUN apt-get update && apt-get install -y \
    wget \
    apt-transport-https \
    gnupg \
    ca-certificates \
    software-properties-common

# Try installing from official Ubuntu repository
RUN apt-get update && apt-get install -y python3-graph-tool || true

# Check if graph-tool was installed from Ubuntu repo
RUN if ! dpkg -s python3-graph-tool >/dev/null 2>&1; then \
    echo "Installing graph-tool from upstream repository..." && \
    wget https://downloads.skewed.de/skewed-keyring/skewed-keyring_1.1_all_jammy.deb && \
    dpkg -i skewed-keyring_1.1_all_jammy.deb && \
    echo "deb [signed-by=/usr/share/keyrings/skewed-keyring.gpg] https://downloads.skewed.de/apt jammy main" > \
    /etc/apt/sources.list.d/skewed.list && \
    apt-get update && \
    apt-get install -y python3-graph-tool; \
    fi

# Install additional dependencies
RUN apt-get install -y --no-install-recommends \
    python3-distutils \
    python3-setuptools \
    python3-dev \
    build-essential \
    pkg-config \
    libcairo2-dev \
    python3-pip \
    python3-venv \
    python3-decorator \
    python3-cairo \
    git \
    cmake \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:/app:$PATH"
ENV PYTHONPATH=/app

RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir torch==1.13.1 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install --no-cache-dir torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cpu.html \
    && pip3 install --no-cache-dir torch-geometric==2.2.0

# Install cython and networkit first
RUN pip3 install cython networkit

# Install project dependencies (fix numpy issue)
RUN pip3 install --no-cache-dir --prefer-binary "numpy>=1.21,<1.24"
COPY ./src/requirements.txt /app/requirements.txt
RUN pip3 install -U --no-cache-dir -r /app/requirements.txt
RUN pip3 cache purge

# Install Docker Compose v2 (for ARM, e.g. aarch64)
RUN apt-get update && apt-get install -y curl && \
    mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -SL https://github.com/docker/compose/releases/download/v2.18.1/docker-compose-linux-aarch64 \
    -o /usr/local/lib/docker/cli-plugins/docker-compose && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

RUN ln -s /usr/local/lib/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose


# Clone and install CABAM Graph Generation tools
RUN git clone https://github.com/snap-research/cabam-graph-generation.git ./src/cabam_graph_generation/
WORKDIR /src/cabam_graph_generation/
RUN pip3 install .
WORKDIR /

# Copy project files
COPY ./src /app

# Install Apache Beam SDK
COPY --from=apache/beam_python3.10_sdk /opt/apache/beam /opt/apache/beam

# Set the entrypoint to Apache Beam SDK worker launcher.
ENTRYPOINT [ "/opt/apache/beam/boot" ]
