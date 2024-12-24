from pathlib import Path

import fastapi
import fastapi.staticfiles
import modal

app = modal.App("got-ocr-model-hf-frontend-v")

web_app = fastapi.FastAPI()


@web_app.post("/parse")
async def parse(request: fastapi.Request):
    parse_receipt = modal.Function.lookup(
        # Name of the backend app
        "got_ocr_volume_backend-v", "parse_receipt"

    )

    form = await request.form()
    receipt = await form["receipt"].read()  # type: ignore
    call = parse_receipt.spawn(receipt)
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
        print(result)
        print(type(result))
        print("I am here")
    except TimeoutError:
        return fastapi.responses.JSONResponse(content="", status_code=202)

    return result


image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]==0.115.4"
)


local_assets_path = Path(__file__).parent / "frontend"
image = image.add_local_dir(local_assets_path, remote_path="/assets")


@app.function(image=image)
@modal.asgi_app()
def wrapper():
    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )
    return web_app

