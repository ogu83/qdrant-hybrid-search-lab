import logging
import uvicorn
from fastapi import FastAPI
from .api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Qdrant Production RAG", version="1.0.0")
app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("ep3_production_rag.main:app", host="0.0.0.0", port=8000, reload=True)
