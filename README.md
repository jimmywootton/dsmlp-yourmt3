# DSMLP image: scipy-ml plus YourMT3+, Demucs, spaces, Git LFS

Base image: `ghcr.io/ucsd-ets/scipy-ml-notebook:2026.4-stable`.

CUDA, PyTorch, and TensorFlow are the versions from that image. A newer CUDA toolkit is not installed, matching the [DSMLP CUDA notes](https://github.com/ucsd-ets/datahub-example-notebook/blob/main/cuda.md).

| Tool | Where |
| --- | --- |
| Jupyter, TensorFlow, PyTorch (CUDA 12.6) | Default kernel, same as scipy-ml |
| Demucs (`demucs`, default `htdemucs` weights cached) | Default kernel and YourMT3+ venv |
| Hugging Face `spaces` | Default kernel and YourMT3+ venv |
| Git LFS | `git lfs` on the system (`git lfs install --system`) |
| YourMT3+ | `/opt/yourmt3`, kernel **Python (YourMT3+)**, command `yourmt3` |

YourMT3+ is pinned to `transformers==4.45.1` and `numpy==1.26.4` in `/opt/yourmt3-venv`. Those pins are not applied to the scipy-ml environment. The default checkpoint baked in is **YPTF.MoE+Multi (noPS)**.

DSMLP mounts over `/home/jovyan`, so the repo, venv, and model cache are under `/opt`.

## Publish

Push this directory to a public GitHub repository. Actions tags the image from the branch name, for example `ghcr.io/<user>/dsmlp-yourmt3:main`. The package must be public.

## Launch

```bash
ssh <user>@dsmlp-login.ucsd.edu
launch-scipy-ml.sh -i ghcr.io/<user>/dsmlp-yourmt3:main -P Always -g 1
```

Drop `-P Always` after the node has the image. Add `-n <node>` to reuse a node that already pulled it.

## Use

```bash
yourmt3 -i /path/to/song.mp3 -o /path/to/song.midi
demucs /path/to/song.mp3
git lfs version
```

In Jupyter, select the **Python (YourMT3+)** kernel before importing the transcription code. The default kernel still has `demucs` and `spaces`.
