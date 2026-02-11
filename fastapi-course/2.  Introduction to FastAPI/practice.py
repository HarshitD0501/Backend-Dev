from fastapi import FastAPI

halua = FastAPI()

@halua.get('/')
def add_integers(a: int, b: int) -> int:
    return a + b


