from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Saved along with Tf-Idf vectorizer. Required to load the serialized vectorizer
def placeholder(x):
    return x


from modcom.ml import predict_comment
from modcom.api import api_router
from modcom.config import get_app_settings


app = FastAPI()
settings = get_app_settings()

Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.include_router(api_router, prefix="/api")


@app.get("/")
def index(request: Request):
    ctx = {"request": request, "models": settings.available_models}
    return templates.TemplateResponse("form.jinja2", context=ctx)


@app.post("/")
def results(
    request: Request, comment: str = Form(...), model: str = Form(...)
):
    result = predict_comment(comment, model)
    ctx = {"request": request, "msg": result.msg}
    response = templates.TemplateResponse("result.jinja2", context=ctx)

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
