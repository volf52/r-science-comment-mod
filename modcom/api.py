from fastapi import APIRouter, Form, Request
from .config import get_api_settings
from .ml import TextPreprocessor
from .models import ModelLoader

router = APIRouter()
settings = get_api_settings()
text_preproc = TextPreprocessor()

model_loader = ModelLoader(settings)
vectorizer = model_loader.vectorizer


@router.get("/hello", tags=["test"])
async def hello_there():
    return {"msg": "Hello there"}


@router.get("/hello/{name}", tags=["test"])
async def hello_user(name: str):
    return {"msg": f"Hello {name}"}


@router.post("/ml", tags=["form"])
async def ml_dodo(req: Request, comment: str = Form(...), model: str = Form(...)):
    clf = model_loader.get_model(model)

    if clf is None:
        return {'msg': 'No such model exists. Check your `model` key'}

    pred = clf.predict(comment)
    return {"msg": f'content is "{comment}"', "preds": pred}
