from fastapi import APIRouter, HTTPException

from api.schemas import NarrateRequest, NarrateResponse
from src.llm import llm_client
from src.config import PERSONA_MAP

router = APIRouter(tags=["persona"])


@router.post("/persona/narrate", response_model=NarrateResponse)
async def narrate_persona(req: NarrateRequest):
    persona = req.persona
    if persona not in PERSONA_MAP.values() and persona not in PERSONA_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown persona: {persona}")

    result = llm_client.generate_narrative(persona, req.profile)
    return NarrateResponse(
        persona=result["persona"],
        narrative=result["narrative"],
        model_used=result["model_used"],
    )


@router.get("/persona/narrate/all", response_model=list[NarrateResponse])
async def narrate_all_personas():
    results = []
    for persona in PERSONA_MAP.values():
        result = llm_client.generate_narrative(persona)
        results.append(NarrateResponse(
            persona=result["persona"],
            narrative=result["narrative"],
            model_used=result["model_used"],
        ))
    return results
