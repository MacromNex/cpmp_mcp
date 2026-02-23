FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/macronex/cpmp_mcp"
LABEL org.opencontainers.image.description="Cyclic Peptide Membrane Permeability prediction using MAT deep learning"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda packages
RUN conda install -y -c conda-forge \
    pandas=2.2.3 scikit-learn=1.6.1 matplotlib seaborn rdkit=2024.3.2 && \
    conda clean -afy

# Pip dependencies
RUN pip install --no-cache-dir \
    numpy scipy fastmcp loguru

# Clone CPMP repo for model checkpoints
RUN git clone https://github.com/panda1103/CPMP.git /app/repo/CPMP || true

# Copy MCP server source
COPY --chmod=755 src/ src/

# Create writable directories for jobs/results
RUN mkdir -p /app/jobs /app/results && chmod 777 /app /app/jobs /app/results

ENV NVIDIA_CUDA_END_OF_LIFE=0
ENTRYPOINT []
CMD ["python", "src/server.py"]
