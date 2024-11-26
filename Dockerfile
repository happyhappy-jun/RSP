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

# Clone repository
RUN git clone https://github.com/happyhappy-jun/RSP.git /workspace && \
    cd /workspace && \
    git checkout master

# Create conda environment
RUN conda create -n rsp python=3.9.12 -y && \
    conda run -n rsp conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia && \
    conda run -n rsp pip install -r /workspace/requirements.txt

# Set up working directory
WORKDIR /workspace

# Create directory for secrets
RUN mkdir -p /workspace/.secrets

# Activate conda environment by default
SHELL ["conda", "run", "-n", "rsp", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]
