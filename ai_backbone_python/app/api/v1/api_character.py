# app/api/v1/api_character.py
"""
캐릭터 정보 관리를 위한 API 엔드포인트
(선택사항: 관리/디버깅 용도)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

from app.core.character_store import character_store

logger = logging.getLogger("api_character")
router = APIRouter()


# =========================
# DTO 정의
# =========================
class CharacterInfoResponse(BaseModel):
    access_id: str
    character_description: Optional[str] = None
    exists: bool


class CharacterStatsResponse(BaseModel):
    total_characters: int
    access_ids: List[str]


class CharacterSetRequest(BaseModel):
    access_id: str
    character_description: str


class CharacterDeleteRequest(BaseModel):
    access_id: str


# =========================
# API 엔드포인트
# =========================
@router.get("/character/{access_id}", response_model=CharacterInfoResponse)
def get_character_info(access_id: str):
    """
    특정 access_id의 캐릭터 정보 조회
    """
    logger.info("[CHARACTER_API] Get character | access_id=%s", access_id)

    character = character_store.get_character(access_id)
    exists = character_store.has_character(access_id)

    return CharacterInfoResponse(
        access_id=access_id,
        character_description=character,
        exists=exists
    )


@router.get("/character/stats/all", response_model=CharacterStatsResponse)
def get_character_stats():
    """
    전체 캐릭터 저장소 통계 조회
    """
    logger.info("[CHARACTER_API] Get stats")

    stats = character_store.get_stats()

    return CharacterStatsResponse(
        total_characters=stats["total_characters"],
        access_ids=stats["access_ids"]
    )


@router.post("/character/set")
def set_character_info(req: CharacterSetRequest):
    """
    캐릭터 정보 수동 설정 (관리용)
    """
    logger.info("[CHARACTER_API] Set character | access_id=%s", req.access_id)

    if not req.character_description or not req.character_description.strip():
        raise HTTPException(status_code=400, detail="Character description cannot be empty")

    character_store.set_character(req.access_id, req.character_description)

    return {
        "success": True,
        "access_id": req.access_id,
        "message": "Character information saved successfully"
    }


@router.delete("/character/delete")
def delete_character_info(req: CharacterDeleteRequest):
    """
    캐릭터 정보 삭제
    """
    logger.info("[CHARACTER_API] Delete character | access_id=%s", req.access_id)

    success = character_store.remove_character(req.access_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Character not found for access_id: {req.access_id}")

    return {
        "success": True,
        "access_id": req.access_id,
        "message": "Character information deleted successfully"
    }


@router.post("/character/clear-all")
def clear_all_characters():
    """
    모든 캐릭터 정보 삭제 (주의: 관리용)
    """
    logger.warning("[CHARACTER_API] Clear all characters requested")

    character_store.clear_all()

    return {
        "success": True,
        "message": "All character information cleared"
    }