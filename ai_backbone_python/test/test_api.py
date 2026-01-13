import requests
import json
import threading
import time

BASE_URL = "http://localhost:8000/api/v1"   # EC2 IP

HEADERS = {"Content-Type": "application/json"}


USERS = [
    {
        "access_id": "userA",
        "character": "빨간 머리의 활발한 소년, 운동복 착용, 밝은 표정",
        "first_prompt": "소년이 운동장에서 공을 차고 있다",
        "second_prompt": "소년이 친구와 하이파이브를 한다"
    },
    {
        "access_id": "userB",
        "character": "안경 쓴 조용한 소녀, 책을 들고 있음, 교복 착용",
        "first_prompt": "소녀가 도서관에서 책을 읽고 있다",
        "second_prompt": "소녀가 창가에서 생각에 잠겨 있다"
    },
    {
        "access_id": "userC",
        "character": "파란 머리의 로봇 소년, 미래형 수트, LED 눈",
        "first_prompt": "로봇 소년이 도시 위를 날고 있다",
        "second_prompt": "로봇 소년이 에너지 빔을 발사한다"
    }
]


def run_user(user):
    access_id = user["access_id"]

    # 1️⃣ 캐릭터 등록
    payload1 = {
        "access_id": access_id,
        "original_content": user["first_prompt"],
        "is_slang": False,
        "access_id_character": user["character"]
    }

    print(f"\n[{access_id}] 1차 요청 전송")
    r1 = requests.post(f"{BASE_URL}/image/generate", headers=HEADERS, json=payload1, timeout=120)
    print(f"[{access_id}] Status: {r1.status_code}")
    print(json.dumps(r1.json(), ensure_ascii=False))

    time.sleep(1)  # 약간의 시간 차

    # 2️⃣ 캐릭터 자동 적용 요청
    payload2 = {
        "access_id": access_id,
        "original_content": user["second_prompt"],
        "is_slang": False
    }

    print(f"[{access_id}] 2차 요청 전송")
    r2 = requests.post(f"{BASE_URL}/image/generate", headers=HEADERS, json=payload2, timeout=120)
    print(f"[{access_id}] Status: {r2.status_code}")
    print(json.dumps(r2.json(), ensure_ascii=False))


if __name__ == "__main__":
    threads = []

    print("\n===== 3명 동시 접속 테스트 시작 =====\n")

    for user in USERS:
        t = threading.Thread(target=run_user, args=(user,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n===== 테스트 종료 =====")
