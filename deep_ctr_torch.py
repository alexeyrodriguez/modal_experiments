from pathlib import Path

import modal


app = modal.App("ctr_torch_test")


REPO_ROOT = Path(__file__).parent
TARGET = "/root/"

image = (
    modal.Image.debian_slim(python_version="3.12.7")
    .pip_install("deepctr-torch", "pandas")
    .apt_install("git")
    .run_commands(["git clone https://github.com/shenweichen/DeepCTR-Torch.git /root/DeepCTR-Torch"])
)


@app.function(image=image)
def train_criteo():
    import os
    import subprocess
    os.chdir(TARGET + "DeepCTR-Torch/examples")
    subprocess.run(["python", "run_classification_criteo.py"])

@app.local_entrypoint()
def main():
    train_criteo.remote()
