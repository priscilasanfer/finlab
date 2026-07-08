from fastapi import FastAPI

from api.routers import agent, rag, search

app = FastAPI(title="Financial Search API")

app.include_router(rag.router)
app.include_router(search.router)
app.include_router(agent.router)


@app.get("/")
def root():
    return {"status": "online "}
