FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Set up working directory
WORKDIR /workspace

# Copy local files
COPY . /workspace/

# Create conda environment from environment.yml
RUN conda env create -f /workspace/environment.yml && \
    conda clean -afy

# Install package in development mode
RUN conda run -n rsp pip install -e ".[dev]"

# Set up working directory
WORKDIR /workspace

# Create directory for secrets
RUN mkdir -p /workspace/.secrets

# Activate conda environment by default
SHELL ["conda", "run", "-n", "rsp", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]
