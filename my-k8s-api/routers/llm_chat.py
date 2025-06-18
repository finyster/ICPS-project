# app/routers/llm_chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

#from services.llm_assistant import chat_with_llm
from services.llm_assistant import chat_with_llm

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    user_message: str
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    assistant: str
    history: List[dict]

@router.post("/llm_chat", response_model=ChatResponse)
def llm_chat(req: ChatRequest):
    try:
        result = chat_with_llm(req.user_message, req.history or [])
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LLM chat failed: %s", e)
        raise HTTPException(status_code=500, detail="LLM backend error")
