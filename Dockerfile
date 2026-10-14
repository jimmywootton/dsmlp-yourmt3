# DSMLP notebook based on the current scipy-ml image.
# CUDA, PyTorch, and TensorFlow stay on the versions UCSD ships in that image.
# https://github.com/ucsd-ets/datahub-example-notebook
# https://github.com/ucsd-ets/datahub-example-notebook/blob/main/cuda.md
#
# YourMT3+ needs transformers 4.45.1 and numpy 1.26. Those pins conflict with
# the scipy-ml TensorFlow stack, so transcription lives in /opt/yourmt3-venv.
# The default Jupyter kernel is unchanged. Pick "Python (YourMT3+)" to
# transcribe. Demucs, the Hugging Face `spaces` package, and Git LFS are on
# the default PATH as well.
#
# DSMLP replaces /home/jovyan with the user volume, so code, the venv, and
# model caches live under /opt.

ARG BASE_CONTAINER=ghcr.io/ucsd-ets/scipy-ml-notebook:2026.4-stable
FROM ${BASE_CONTAINER}

LABEL maintainer="DSMLP custom audio image"

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        ffmpeg \
        sox \
        libsndfile1 \
        lame \
        build-essential \
        ca-certificates \
        curl && \
    git lfs install --system && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/opt/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/opt/cache/huggingface/hub \
    TORCH_HOME=/opt/cache/torch \
    XDG_CACHE_HOME=/opt/cache \
    WANDB_MODE=disabled \
    YOURMT3_HOME=/opt/yourmt3

RUN mkdir -p /opt/cache/huggingface /opt/cache/torch /opt/cache/uv /opt/yourmt3 && \
    fix-permissions /opt/cache

# Default kernel: Demucs and `import spaces`, without moving torch or numpy.
# UV_CACHE_DIR is explicit because XDG_CACHE_HOME points at /opt/cache, and
# jovyan cannot initialize a cache there until fix-permissions has run.
USER jovyan
RUN UV_CACHE_DIR=/tmp/uv-cache uv pip freeze > /tmp/scipy-ml-constraints.txt && \
    UV_CACHE_DIR=/tmp/uv-cache uv pip install --system --no-deps "demucs==4.0.1" && \
    UV_CACHE_DIR=/tmp/uv-cache uv pip install --system \
        --constraint /tmp/scipy-ml-constraints.txt \
        "dora-search>=0.1.12" \
        "diffq>=0.2.1" \
        einops \
        "hydra-colorlog>=1.1" \
        "hydra-core>=1.1" \
        "julius>=0.2.3" \
        "lameenc>=1.2" \
        "openunmix>=1.2.1" \
        pyyaml \
        tqdm \
        treetable \
        spaces && \
    UV_CACHE_DIR=/tmp/uv-cache uv cache clean && \
    rm -rf /tmp/uv-cache /tmp/scipy-ml-constraints.txt

USER root

RUN python3 -m venv /opt/yourmt3-venv && \
    /opt/yourmt3-venv/bin/pip install --no-cache-dir --upgrade pip

# Same CUDA 12.6 wheels as scipy-ml. A newer toolkit is not supported on DSMLP.
RUN /opt/yourmt3-venv/bin/pip install --no-cache-dir \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu126

RUN git clone --depth 1 https://github.com/mimbres/AMT-challenge.git /opt/yourmt3

# Best AMT Challenge 2025 checkpoint. The field in cfg_local.py is
# default_model_name (the published note says default_mode_name).
RUN python3 - <<'PY'
from pathlib import Path
path = Path("/opt/yourmt3/cfg_local.py")
text = path.read_text()
old = 'default_model_name = "YMT3+"'
new = 'default_model_name = "YPTF.MoE+Multi (noPS)"'
if old not in text:
    raise SystemExit("default model line missing from cfg_local.py")
path.write_text(text.replace(old, new, 1))
PY

# Drop torch and torchaudio so pip does not replace the CUDA 12.6 wheels.
# Drop the cu113 extra index that upstream requirements.txt points at.
RUN python3 - <<'PY'
from pathlib import Path
skip = {"torch", "torchaudio"}
kept = []
for line in Path("/opt/yourmt3/amt/requirements.txt").read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("--"):
        continue
    if stripped.startswith("git+"):
        kept.append(stripped)
        continue
    name = stripped.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip().lower()
    if name not in skip:
        kept.append(stripped)
Path("/tmp/yourmt3-requirements.txt").write_text("\n".join(kept) + "\n")
PY

RUN /opt/yourmt3-venv/bin/pip install --no-cache-dir -r /tmp/yourmt3-requirements.txt && \
    /opt/yourmt3-venv/bin/pip install --no-cache-dir -e /opt/yourmt3/amt --no-deps && \
    /opt/yourmt3-venv/bin/pip install --no-cache-dir \
        "transformers==4.45.1" \
        "numpy==1.26.4" \
        spaces \
        gradio \
        gradio-log \
        yt-dlp \
        python-dotenv \
        awscli \
        ipykernel && \
    printf '%s\n' \
        'torch==2.8.0' \
        'torchaudio==2.8.0' \
        'numpy==1.26.4' \
        'transformers==4.45.1' \
        > /tmp/yourmt3-constraints.txt && \
    /opt/yourmt3-venv/bin/pip install --no-cache-dir --no-deps "demucs==4.0.1" && \
    /opt/yourmt3-venv/bin/pip install --no-cache-dir \
        --constraint /tmp/yourmt3-constraints.txt \
        "dora-search>=0.1.12" \
        "diffq>=0.2.1" \
        "hydra-colorlog>=1.1" \
        "hydra-core>=1.1" \
        "julius>=0.2.3" \
        "lameenc>=1.2" \
        "openunmix>=1.2.1" \
        pyyaml \
        tqdm \
        treetable && \
    rm -f /tmp/yourmt3-requirements.txt /tmp/yourmt3-constraints.txt

RUN /opt/yourmt3-venv/bin/python -m ipykernel install \
        --prefix=/opt/conda \
        --name yourmt3 \
        --display-name "Python (YourMT3+)"

COPY yourmt3 /usr/local/bin/yourmt3
RUN chmod 755 /usr/local/bin/yourmt3

# Public checkpoint, about 560MB. Other models stay on S3 until requested
# (aws s3 cp --no-sign-request, awscli is in the YourMT3+ venv).
RUN mkdir -p /opt/yourmt3/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints && \
    curl -fL --retry 5 -o \
        /opt/yourmt3/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt \
        https://amt-deploy-public.s3.amazonaws.com/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt

# Cache the default Demucs weights outside the home directory DSMLP mounts over.
RUN /opt/yourmt3-venv/bin/python - <<'PY'
from demucs.pretrained import get_model
get_model("htdemucs")
PY

RUN /opt/yourmt3-venv/bin/python - <<'PY'
import demucs
import mockmt3
import spaces
import torch
import transformers
assert transformers.__version__ == "4.45.1", transformers.__version__
print("yourmt3-venv", torch.__version__, "cuda", torch.version.cuda)
PY
RUN python3 - <<'PY'
import demucs
import spaces
import torch
print("scipy-ml", torch.__version__, "cuda", torch.version.cuda)
PY
RUN git lfs version && command -v demucs && command -v yourmt3

RUN fix-permissions /opt/yourmt3 && \
    fix-permissions /opt/yourmt3-venv && \
    fix-permissions /opt/cache && \
    fix-permissions /opt/conda/share/jupyter

WORKDIR /home/jovyan
USER jovyan
