# app/core/character_store.py
"""
access_id별 웹툰 캐릭터 정보를 메모리에 저장하고 관리하는 모듈
"""
import logging
from typing import Dict, Optional
from threading import RLock

logger = logging.getLogger("character_store")


class CharacterStore:
    """
    access_id와 캐릭터 설명을 매핑하여 저장하는 싱글톤 클래스
    """
    _instance = None
    _lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._character_map: Dict[str, str] = {}
        self._access_lock = RLock()
        self._initialized = True
        logger.info("[CHARACTER_STORE] Initialized")

    def set_character(self, access_id: str, character_description: str) -> None:
        """
        access_id에 대한 캐릭터 설명을 저장

        Args:
            access_id: 사용자 식별자
            character_description: 캐릭터 설명 (예: "파란 머리의 소녀, 큰 눈, 학교 교복")
        """
        if not access_id or not access_id.strip():
            logger.warning("[CHARACTER_STORE] Invalid access_id provided")
            return

        if not character_description or not character_description.strip():
            logger.warning("[CHARACTER_STORE] Empty character description for access_id=%s", access_id)
            return

        with self._access_lock:
            self._character_map[access_id] = character_description.strip()
            logger.info(
                "[CHARACTER_STORE] Character saved | access_id=%s, description_length=%d",
                access_id, len(character_description)
            )

    def get_character(self, access_id: str) -> Optional[str]:
        """
        access_id에 저장된 캐릭터 설명을 조회

        Args:
            access_id: 사용자 식별자

        Returns:
            캐릭터 설명 문자열, 없으면 None
        """
        if not access_id or not access_id.strip():
            return None

        with self._access_lock:
            character = self._character_map.get(access_id)
            if character:
                logger.debug("[CHARACTER_STORE] Character retrieved | access_id=%s", access_id)
            return character

    def has_character(self, access_id: str) -> bool:
        """
        access_id에 캐릭터가 등록되어 있는지 확인

        Args:
            access_id: 사용자 식별자

        Returns:
            캐릭터 존재 여부
        """
        if not access_id:
            return False

        with self._access_lock:
            return access_id in self._character_map

    def remove_character(self, access_id: str) -> bool:
        """
        access_id의 캐릭터 정보를 삭제

        Args:
            access_id: 사용자 식별자

        Returns:
            삭제 성공 여부
        """
        if not access_id:
            return False

        with self._access_lock:
            if access_id in self._character_map:
                del self._character_map[access_id]
                logger.info("[CHARACTER_STORE] Character removed | access_id=%s", access_id)
                return True
            return False

    def get_all_access_ids(self) -> list:
        """
        등록된 모든 access_id 목록 반환

        Returns:
            access_id 리스트
        """
        with self._access_lock:
            return list(self._character_map.keys())

    def clear_all(self) -> None:
        """
        모든 캐릭터 정보 삭제 (테스트/관리 용도)
        """
        with self._access_lock:
            count = len(self._character_map)
            self._character_map.clear()
            logger.warning("[CHARACTER_STORE] All characters cleared | count=%d", count)

    def get_stats(self) -> Dict:
        """
        저장소 통계 정보 반환

        Returns:
            통계 딕셔너리
        """
        with self._access_lock:
            return {
                "total_characters": len(self._character_map),
                "access_ids": list(self._character_map.keys())
            }


# 싱글톤 인스턴스 생성
character_store = CharacterStore()