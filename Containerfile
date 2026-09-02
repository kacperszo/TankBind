FROM docker.io/mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -c "import torch, torch_geometric, torchdrug; from rdkit import Chem; import atom3d; print('env ok')"

# crypt.h is included by Python.h but missing from the conda python3.8 headers.
# libxcrypt-dev provides it so the torchdrug C++ extension can compile.
USER root
RUN apt-get update && apt-get install -y libcrypt-dev && \
    ln -s /usr/include/crypt.h /opt/conda/include/python3.8/crypt.h && \
    rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# torchdrug compiles C++ extensions on import and needs somewhere to write them. The rootfs
# is read-only by design, so point it at the tmpfs the runner mounts.
ENV TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
ENV HOME=/tmp

WORKDIR /work
COPY --chown=$MAMBA_USER:$MAMBA_USER . /work

# torchdrug JIT-compiles a C++ extension (sparse_coo_tensor_unsafe) on first use.
# Pre-warming here bakes the compiled cache into the image so container startup
# doesn't stall for ~10s on every run.
RUN python -c "import torch; from torchdrug import utils; utils.sparse_coo_tensor(torch.zeros(2,1,dtype=torch.long), torch.ones(1), (2,2))"
