import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_with_character():
    """캐릭터 정보를 포함한 첫 요청"""
    payload = {
        "access_id": "user123",
        "original_content": "캐릭터가 환하게 웃으면서 손을 흔들고 있어",
        "is_slang": False,
        "access_id_character": "파란 머리의 귀여운 소녀, 큰 눈동자, 하얀색 학교 교복 착용, 귀여운 미소"
    }

    print("=" * 60)
    print("테스트 1: 캐릭터 등록 요청")
    print("=" * 60)
    print(f"Request: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        response = requests.post(
            f"{BASE_URL}/image/generate",
            json=payload,
            timeout=60
        )

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_without_character():
    """캐릭터 정보 없이 두 번째 요청 (자동 적용 테스트)"""
    payload = {
        "access_id": "user123",
        "original_content": "캐릭터가 학교 복도를 신나게 달리고 있어. 배경은 밝은 학교 복도",
        "is_slang": False
    }

    print("\n" + "=" * 60)
    print("테스트 2: 캐릭터 자동 적용 요청")
    print("=" * 60)
    print(f"Request: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        response = requests.post(
            f"{BASE_URL}/image/generate",
            json=payload,
            timeout=60
        )

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_get_character():
    """저장된 캐릭터 정보 조회 (관리 API가 있는 경우)"""
    access_id = "user123"

    print("\n" + "=" * 60)
    print("테스트 3: 캐릭터 정보 조회")
    print("=" * 60)

    try:
        response = requests.get(
            f"{BASE_URL}/character/{access_id}",
            timeout=10
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        return response.json()
    except Exception as e:
        print(f"캐릭터 조회 API가 없거나 오류 발생: {e}")
        return None


if __name__ == "__main__":
    # 테스트 1: 캐릭터 등록
    result1 = test_with_character()

    if result1:
        # 테스트 2: 캐릭터 자동 적용
        result2 = test_without_character()

        # 테스트 3: 저장된 캐릭터 조회 (선택사항)
        test_get_character()

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)