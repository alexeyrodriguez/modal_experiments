from pathlib import Path

import modal


app = modal.App("ctr_torch_test")

# The BARS benchmark will not fork because BARS depends on an older version of FuxiCTR
# that needs an old version of python that modal does not support.
# However, this test is based on the BARS benchmark.

image = (
    modal.Image.debian_slim(python_version="3.11.7")
    .apt_install("git")
    .pip_install("tqdm")
    .pip_install("numpy==1.25.2", "scikit-learn==1.4")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu124",  # tested with torch-2.6.0.dev20241120
    )
    .run_commands(["pip install git+https://github.com/reczoo/FuxiCTR.git"])
    .run_commands(["git clone https://github.com/reczoo/FuxiCTR.git /root/FuxiCTR"])
    .run_commands(["git clone https://github.com/reczoo/Datasets.git /root/Datasets"])
    .run_commands(["git clone https://github.com/reczoo/BARS.git /root/BARS"])
    .add_local_dir("criteo_deepfm_config", "/configs/criteo_deepfm_config")
    .add_local_dir("criteo_dcnv2_config", "/configs/criteo_dcnv2_config")
    .add_local_dir("criteo_dnn_config", "/configs/criteo_dnn_config")
)

data = modal.Volume.from_name("criteo", create_if_missing=True)

@app.function(volumes={"/data": data}, image=image, memory=90*1024, timeout=60*60)
def download_criteo_data():
    import os
    import subprocess
    os.chdir("/data")
    if not os.path.exists("./criteo"):
        subprocess.run(["python", "/root/Datasets/Criteo/Criteo_x1/download_criteo_x1.py"])
        os.chdir("./criteo")
        print("Starting conversion...")
        subprocess.run(["python", "/root/Datasets/Criteo/Criteo_x1/convert_criteo_x1.py"])

@app.function(image=image)
def train_basic():
    import os
    import subprocess
    # Fix incorrect way of referring to data
    subprocess.run(["ln", "-s", "-f", "/root/FuxiCTR/data", "/root/FuxiCTR/model_zoo/data"])
    subprocess.run([
        "python",
            "/root/FuxiCTR/model_zoo/DeepFM/DeepFM_torch/run_expid.py",
    ])

@app.function(volumes={"/data": data}, image=image, memory=90*1024, timeout=10*60*60, gpu=f"a100")
def train_deepfm():
    import subprocess
    subprocess.run([
        "python",
            "/root/FuxiCTR/model_zoo/DeepFM/DeepFM_torch/run_expid.py",
            "--config", "/configs/criteo_deepfm_config",
            "--expid", "DeepFM_criteo_x1",
            "--gpu", "0"
    ])

@app.function(volumes={"/data": data}, image=image, memory=90*1024, timeout=10*60*60, gpu=f"a100")
def train_dcnv2():
    import subprocess
    subprocess.run([
        "python",
            "/root/FuxiCTR/model_zoo/DCNv2/run_expid.py",
            "--config", "/configs/criteo_dcnv2_config",
            "--expid", "DCNv2_criteo_x1",
            "--gpu", "0"
    ])

@app.function(volumes={"/data": data}, image=image, memory=90*1024, timeout=10*60*60, gpu=f"a100")
def train_dnn():
    import subprocess
    subprocess.run([
        "python",
            "/root/FuxiCTR/model_zoo/DNN/DNN_torch/run_expid.py",
            "--config", "/configs/criteo_dnn_config",
            "--expid", "DNN_criteo_x1",
            "--gpu", "0"
    ])

@app.local_entrypoint()
def main():
    download_criteo_data.remote()
    train_dnn.remote()
    train_deepfm.remote()
    train_dcnv2.remote()
