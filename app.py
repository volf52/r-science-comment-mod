from fastapi import FastAPI
from fastapi.responses import FileResponse


# Saved along with Tf-Idf vectorizer. Requires to function
def placeholder(x):
    return x


from modcom.api import router

app = FastAPI()

app.include_router(router, prefix="/api")


@app.get("/")
def index():
    return FileResponse("./frontend/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
