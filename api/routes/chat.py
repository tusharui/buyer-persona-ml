from fastapi import APIRouter

from api.schemas import ChatRequest, ChatResponse
from src.rag import rag_chatbot

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not rag_chatbot._initialized:
        rag_chatbot.initialize()
    result = rag_chatbot.query(req.query)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
    )
