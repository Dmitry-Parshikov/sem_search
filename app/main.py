from fastapi import FastAPI

app = FastAPI(title="sem_search")


@app.get("/health")
def health():
    return {"status": "ok"}
