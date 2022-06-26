import base64
import json
from fastapi import FastAPI
from fastapi.responses import Response, HTMLResponse, RedirectResponse
import uvicorn

from model import Model, CLASSES_TO_IDX

app = FastAPI()
model = Model()


@app.get("/")
async def root():
    # return {"message": "Hello World"}
    return RedirectResponse("/docs")


@app.get(
    "/image", response_class=HTMLResponse,
)
async def get_image():
    result = model.get_next_image()
    if result is None:
        return HTMLResponse(
            content="""
            <html>
                <head>
                    <title>nope</title>
                </head>
                <body>
                    <h1>No images left to annotate. Train the new annotations!</h1>
                </body>
            </html>
            """
        )
    else:
        image_bytes, image_id, answer = result
        return HTMLResponse(
            content=f"""
            <html>
                <head>
                    <title>image to annotate</title>
                </head>
                <body>
                    <h1>Image id: {image_id}</h1>
                    <p>answer: {answer}</p>
                    <img src='data:image/png;base64, {base64.b64encode(image_bytes).decode("utf-8")}' />
                    <p>Annotation options: {list(CLASSES_TO_IDX.keys())}</p>
                </body>
            </html>
            """
        )


@app.get("/evaluate")
def evaluate():
    return {"eval perf": str(model.perform_eval())}


@app.get("/annotations")
async def get_annos():
    return model.annotations


@app.put("/annotate/")
async def record_anno(image_id: int, annotation: str):
    invalid_format_response = Response(
        content=f"{image_id} or {annotation} is not a valid annotation format. Annotation has not been recorded.",
        status_code=404,
    )

    try:
        if (
            int(image_id) > model.accepted_id_max
            or int(image_id) < model.accepted_id_min
        ):
            invalid_id_response = Response(
                content=f"{image_id} is not a valid image id. Annotation has not been recorded.",
                status_code=404,
            )
            return invalid_id_response

        if annotation not in CLASSES_TO_IDX:
            return Response(
                content=f"Unexpected annotation: {annotation}. Annotation has not been recorded.",
                status_code=404,
            )

    except Exception:
        return invalid_format_response

    model.annotations[int(image_id)] = annotation

    return Response(
        content=f"image {image_id} successfully annotated as {annotation}",
        status_code=200,
    )


@app.get("/train_annos")
def train_annos():
    acc: float = model.train_annotations()
    return {"eval set acc": str(acc)}


@app.post("/commit")
def commit_checkpoints(name: str):
    model.commit_checkpoint(name=name)
    return f"checkpoint {name} saved successfully."


@app.get("/list")
def list_checkpoints():
    result = model.list_checkpoints()
    return "no checkpoints found" if not result else result


@app.post("/load")
def load_checkpoint(name: str):
    model.load_checkpoint(name=name)
    return f"checkpoint {name} loaded successfully."


@app.post("/train_baseline")
def train_baseline():
    model.train_baseline(epochs=1)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=7777, reload=True)
