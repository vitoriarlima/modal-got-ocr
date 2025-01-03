# ---
# deploy: true
# cmd: ["modal", "serve", "got_ocr_webapp.py"]
# ---

# This is a draft for the blogpost. 


# # GOT-OCR Web Application

# This tutorial shows you how to use Modal to deploy a fully serverless
# [React](https://reactjs.org/) + [FastAPI](https://fastapi.tiangolo.com/) application.

# We're going to build a simple "Text to Image" web app that submits OCR transcription
# tasks to a separate Modal app defined in the [Job Queue tutorial](to be updated),
# polls until the task is completed, and displays
# the results. 

# Try it out for yourself [here](https://vitoria--got-ocr-model-hf-frontend-v-wrapper-dev.modal.run ).


# ![document processing interface](./doc_ocr_process.jpg)

# ## Basic Setup

# Let's get the imports out of the way and define an [`App`](https://modal.com/docs/reference/modal.App).


from pathlib import Path

import fastapi
import fastapi.staticfiles
import modal

# Create our Modal app instance with a unique identifier
app = modal.App("got-ocr-model-hf-frontend-v")

# Modal works with any [ASGI](https://modal.com/docs/guide/webhooks#serving-asgi-and-wsgi-apps) or
# [WSGI](https://modal.com/docs/guide/webhooks#wsgi) web framework. Here, we choose to use [FastAPI](https://fastapi.tiangolo.com/).

# Initialize FastAPI for our web application
web_app = fastapi.FastAPI()

# ## API Endpoints

# We need two main endpoints:
# 1. `/parse` - Accepts document images and submits them to our OCR pipeline
# 2. `/result` - Polls for and retrieves the OCR results

# The parse endpoint handles document uploads and initiates OCR processing.
# It connects to our GOT-OCR backend service using Modal's function lookup.


@web_app.post("/parse")
async def parse(request: fastapi.Request):
    parse_receipt = modal.Function.lookup(
        # Connect to our OCR backend service
        # Name of the backend app
        "got_ocr_volume_backend-v", "parse_receipt"

    )
    # Process the uploaded file
    form = await request.form()
    receipt = await form["receipt"].read()  # type: ignore

    # Submit the task to our OCR pipeline
    call = parse_receipt.spawn(receipt)
    return {"call_id": call.object_id}

# The results endpoint handles polling for OCR results.
# It uses the call_id to track and retrieve the processing status.

@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    # Look up the function call using its ID
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        # Get the result of the function call
        result = function_call.get(timeout=0)
        print(result)
        print(type(result))
        print("I am here")
    except TimeoutError:
        # Return 202 status if processing isn't complete
        return fastapi.responses.JSONResponse(content="", status_code=202)

    return result

# ## Environment Setup

# Configure our container image with the necessary dependencies.
# We use a lightweight Debian image with FastAPI installed.

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4"
)

# ## Frontend Integration

# Include our React frontend assets in the deployment.
# The frontend code is stored in the local 'frontend' directory.


local_assets_path = Path(__file__).parent / "frontend"
image = image.add_local_dir(local_assets_path, remote_path="/assets")

# ## Application Wrapper

# Create our main application wrapper that serves both the API
# and static frontend files.

@app.function(image=image)
@modal.asgi_app()
def wrapper():
    # Mount our frontend assets at the root path
    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app


# ## Development

# To run this application locally during development:
# ```shell
# modal serve got_ocr_webapp.py
# ```

# ## Deploy

# To deploy this pipeline, run the following commands in this order:

# 1. Download the model weights:
# ```shell
# modal run download_got.py
# ```

# 2. Deploy the pipeline:
# ```shell
# modal deploy doc_ocr_jobs_got_volume.py
# ```

# 3. Serve the app:
# ```shell
# modal serve doc_ocr_webapp.py
# ```

# That's all!

# If successful, this will print a URL for your app that you can navigate to in
# your browser 🎉 .