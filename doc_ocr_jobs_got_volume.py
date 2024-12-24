# ---
# deploy: False
# ---

# This is a draft for the blogpost. 

# # Scalable Document OCR Pipeline with GOT-OCR

# This tutorial demonstrates how to build a production-ready OCR pipeline using Modal
# and the GOT-OCR model. The pipeline automatically scales based on demand and efficiently
# processes document images to extract formatted text.

# We'll use Modal's cloud infrastructure for deployment and the state-of-the-art GOT-OCR
# [model](https://huggingface.co/stepfun-ai/GOT-OCR2_0). 

# Try it out by deploying this code to Modal and sending document images for processing!

# ![document OCR process](./doc_ocr_process.jpg)

# ## Setup and Imports

# First, we import required packages and set up our Modal app with a persistent volume
# for storing model weights.



import urllib.request
from pathlib import Path
import modal
import os

app = modal.App("got_ocr_volume_backend-v")

# Create a persistent volume to store our model weights
# This ensures we don't need to download them on every run

# Create or get the volume for model weights
volume = modal.Volume.from_name("got-ocr-weights", create_if_missing=True)
MODEL_DIR = Path("/models")

# Model information
HF_MODEL_NAME = "ucaslcl/GOT-OCR2_0"
LOCAL_MODEL_NAME = "GOT"

# ## Container Images

# We need two specialized container images:
# 1. A lightweight image for fast model downloads
# 2. A full inference image with ML dependencies

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

# The inference image contains all required ML libraries
inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        "transformers==4.37.2",
        "tiktoken==0.6.0",
        "verovio==4.3.1",
        "accelerate==0.28.0",
        "huggingface_hub[hf_transfer]",
        "numpy>=1.17.0,<2.0.0",  # Explicitly specify numpy version range
    )
)

# ## Model Download Function

# This function handles downloading and caching of model weights.
# It uses Modal's volume system to persist the weights between runs.


@app.function(
    volumes={MODEL_DIR: volume},
    image=download_image,
)
def download_model():
    from huggingface_hub import snapshot_download
    import shutil
    
    model_path = MODEL_DIR / LOCAL_MODEL_NAME
    
    print(f"Starting GOT-OCR model download to {model_path}")
    print(f"Current directory contents before download: {os.listdir(MODEL_DIR)}")
    
    try:
        # Clear existing model directory if it exists
        if model_path.exists():
            shutil.rmtree(model_path)
        
        # Download the model
        snapshot_download(
            repo_id=HF_MODEL_NAME,
            local_dir=model_path,
        )
        
        print(f"Download completed. Contents of {model_path}:")
        print(os.listdir(model_path))
        
        return True
        
    except Exception as e:
        print(f"Error during model download: {str(e)}")
        raise e

# ## Model Wrapper

# The ModelWrapper class provides a clean interface for model operations.
# It handles model initialization and ensures proper GPU usage.


class ModelWrapper:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        import os
        
        model_path = MODEL_DIR / LOCAL_MODEL_NAME
        print(f"Checking GOT-OCR model path: {model_path}")
        print(f"Directory contents: {os.listdir(model_path)}")
        
        print("Loading GOT-OCR tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),  # Convert Path to string
            trust_remote_code=True,
        )
        
        print("Loading GOT-OCR model...")
        self.model = AutoModel.from_pretrained(
            str(model_path),  # Convert Path to string
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='cuda',
            use_safetensors=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.model = self.model.eval().cuda()
        print("Model loaded successfully")
        return self

# ## OCR Pipeline Handler

# This is our main function that processes images.
# With ``gpu="any"`` and ``retries=3``, it automatically uses GPU and includes retry logic.

@app.function(
    gpu="any",
    image=inference_image,
    volumes={MODEL_DIR: volume},
    retries=3,
)
def parse_receipt(image_bytes: bytes):
    import io
    from PIL import Image
    
    # First verify model is downloaded
    try:
        download_model.remote()
    except Exception as e:
        print(f"Error during model download: {str(e)}")
        raise e
    
    print("GOT-OCR model download verified, initializing wrapper...")
    model_wrapper = ModelWrapper()
    model_wrapper.load_model()
    
    # Convert bytes to PIL Image
    input_img = Image.open(io.BytesIO(image_bytes))
    
    # Save image temporarily
    temp_path = "/tmp/temp_image.jpg"
    input_img.save(temp_path)
    
    formatted_result = model_wrapper.model.chat(
        model_wrapper.tokenizer, 
        temp_path, 
        ocr_type='format'
    )
    
    return formatted_result

# ## Local Testing

# For easier debugging, we can run the pipeline locally:
# `modal run ocr_pipeline.py`

@app.local_entrypoint()
def main():
    # Then run inference
    from pathlib import Path
    receipt_filename = Path(__file__).parent / "receipt.png"
    if receipt_filename.exists():
        with open(receipt_filename, "rb") as f:
            image = f.read()
        print(f"Running GOT-OCR on {f.name}")
    else:
        receipt_url = "https://nwlc.org/wp-content/uploads/2022/01/Brandys-walmart-receipt-8.webp"
        image = urllib.request.urlopen(receipt_url).read()
        print(f"Running GOT-OCR on sample from URL {receipt_url}")
    
    result = parse_receipt.remote(image)
    print("Plain text result:", result["plain_text"])
    if result["formatted"]:
        print("\nFormatted result:", result["formatted"])


# ## Deploy

# To deploy this pipeline:
# ```shell
# modal deploy doc_ocr_jobs_got_volume.py
# ```

# Once deployed, you can serve the app with:
# ```shell
# modal serve doc_ocr_webapp.py
# ```

# This will start the frontend app with which the user can interact.
