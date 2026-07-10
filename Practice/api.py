from fastapi import FastAPI


app = FastAPI(title="Practice FastAPI")


@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}


@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Hello, {name}"}
