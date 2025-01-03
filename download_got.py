# ---
# args: ["--force-download"]
# ---

import modal
from pathlib import Path
import os

# Model information
VOLUME_NAME = "got-ocr-model-hf"
MODEL_DIR = Path(f"/{VOLUME_NAME}")
HF_MODEL_NAME = "ucaslcl/GOT-OCR2_0"

# ## Volume Setup
# Create a persistent volume to store our model weights
# This ensures we don't need to download them on every run

# Create or get the volume for model weights
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# The download image uses Hugging Face's fast download protocol
download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "huggingface_hub[hf_transfer]", 
        "transformers", 
        "numpy>=1.17.0,<2.0.0",  # Explicitly specify numpy version range
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ## Model Download Function

# This function handles downloading and caching of model weights.
# It uses Modal's volume system to persist the weights between runs.

app = modal.App(
    image=download_image,
)

@app.function(
    volumes={MODEL_DIR: volume},
    image=download_image,
)
def download_model(force_download: bool = False):
    from huggingface_hub import snapshot_download

    model_path = MODEL_DIR / HF_MODEL_NAME
    print(f"Model does not exist at {model_path}")
    print(f"Starting GOT-OCR model download to {model_path}")
    print(f"Current directory contents before download: {os.listdir(MODEL_DIR)}")
    snapshot_download(
            repo_id=HF_MODEL_NAME,
            local_dir=model_path,
            force_download=force_download,
        )
    print(f"Download completed. Contents of {model_path}: {os.listdir(MODEL_DIR)}")
    return (f"Download completed. Contents of {model_path}: {os.listdir(MODEL_DIR)}")

# ## Download the Model Weights

# To download the model weights, run the following command:

# ```shell
# modal run download_got.py
# ```

@app.local_entrypoint()
def main(
    force_download: bool = False,
):
    download_model.remote(force_download)
    print(f"Download completed. Finished completing execution of download_got.py")