from fastapi import APIRouter, Form, Request

router = APIRouter()


@router.get("/hello", tags=["test"])
async def hello_there():
    return {"msg": "Hello there"}


@router.get("/hello/{name}", tags=["test"])
async def hello_user(name: str):
    return {"msg": f"Hello {name}"}


@router.post('/ml', tags=['form'])
async def ml_dodo(req: Request, comment: str = Form(...)):
    return {"msg": f'content is "{comment}"'}
