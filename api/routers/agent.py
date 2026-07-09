from fastapi import APIRouter, HTTPException

from api.models.agent import AgentRequest, AgentResponse
from api.routers.search import search_service
from api.services.agent import AgentService

router = APIRouter()

agent_service = AgentService(search_service=search_service)


@router.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest):
    try:
        return await agent_service.analyze(request.query, request.limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
