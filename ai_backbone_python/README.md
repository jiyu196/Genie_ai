# **🧠 Webtoon AI – Python AI Backbone**

**Korean Prompt → Safe Webtoon Prompt → Image Generation → Korean Scene Output**

이 프로젝트는 **한국어 입력 문장**을 받아

1. **비속어·위험어 순화(KoBART)**

2. **웹툰 스타일 프롬프트 생성**

3. **DALL·E 3 이미지 생성**

4. **결과를 다시 한국어 웹툰 문장으로 변환**  
    하는 **AI 파이프라인 서버**입니다.

Spring Boot Gateway → 이 FastAPI 서버 → OpenAI API 구조로 동작합니다.

---

# **📁 디렉터리 구조**

`ai_backbone_python_260112/`  
`├── app/`  
`│   ├── main.py`  
`│   ├── api/`  
`│   │   └── v1/`  
`│   │       ├── api_image.py`  
`│   │       └── api_character.py`  
`│   ├── core/`  
`│   │   ├── character_store.py`  
`│   │   ├── config.py`  
`│   │   └── startup.py`  
`│   ├── model/`  
`│   │   ├── kobart_purifier_stage3_short/`  
`│   │   ├── forbidden_to_clean.csv`  
`│   │   ├── model_loader.py`  
`│   │   ├── purifier.py`  
`│   │   └── validation.py`  
`│   ├── service/`  
`│   │   ├── openai_image_service.py`  
`│   │   ├── post_processor.py`  
`│   │   ├── prompt_builder.py`  
`│   │   ├── refine_service.py`  
`│   │   ├── request_trace.py`  
`│   │   └── translator.py`  
`│   └── test/`  
`│       └── test_api.py`

---

# **🧩 전체 동작 흐름**

`사용자 한국어 문장`  
   `↓`  
`FastAPI (/image/generate)`  
   `↓`  
`KoBART 비속어 순화 (GPU)`  
   `↓`  
`후처리 (validation + 의미보존)`  
   `↓`  
`웹툰 스타일 프롬프트 생성`  
   `↓`  
`OpenAI DALL-E-3`  
   `↓`  
`revised_prompt 번역`  
   `↓`  
`최종 한국어 웹툰 장면 문장 생성`  
   `↓`  
`JSON 응답`

---

# **📌 핵심 파일 설명**

## **1️⃣ `main.py`**

FastAPI 앱의 진입점

역할:

* 앱 생성

* `startup.py` 실행

* `/api/v1` 라우터 등록

---

## **2️⃣ `api/v1/api_image.py`**

**이 프로젝트의 핵심 오케스트레이터**

기능:

* 전체 AI 파이프라인을 **RequestContext 단위로 관리**

* GPU KoBART → OpenAI → Translator → Response 를 트랜잭션으로 묶음

* GPU 추론은 `Semaphore(1)` 로 직렬화

* OpenAI 실패 시도 soft-failure로 처리

주요 단계:

| 단계 | 함수 |
| ----- | ----- |
| 캐릭터 처리 | `step_character()` |
| 비속어 순화 | `step_purify()` |
| 프롬프트 생성 | `step_build_prompt()` |
| 이미지 생성 | `step_openai()` |
| 번역 | `step_translate()` |
| 한국어 문장 구성 | `step_compose_korean_scene()` |

---

## **3️⃣ `core/character_store.py`**

**유저별 캐릭터 상태 저장소**

`access_id → "파란 머리의 귀여운 소녀"`

같은 사용자가 여러 요청을 보내도 캐릭터 일관성이 유지됨.

---

## **4️⃣ `model/model_loader.py`**

KoBART 모델을 디스크에서 로드

환경변수로 모델 경로 변경 가능

---

## **5️⃣ `model/purifier.py`**

KoBART 추론기

입력:`"씨발년이 웃고 있음"`
출력:`"무례한 여자가 웃고 있다"`

---

## **6️⃣ `model/validation.py`**

KoBART 출력 검증

* 빈 문자열

* 반복 토큰

* 특수문자 폭주

* 의미없는 토큰 차단

---

## **7️⃣ `service/post_processor.py`**

KoBART 출력 후처리

기능:

* 과도한 변형 방지

* 의미 보존 체크

* 화이트리스트 보호

* 잘못된 출력 시 원문 fallback

Chris님이 성능 최적화 버전으로 교체함  
 (정규식 캐시, 빠른 단어 겹침 검사)

---

## **8️⃣ `service/prompt_builder.py`**

웹툰 스타일 프롬프트 생성기

예:

`character: "파란 머리 소녀"`  
`scene: "아이들이 뛰어놀고 있다"`

→ DALL-E용 영어 프롬프트로 변환

---

## **9️⃣ `service/openai_image_service.py`**

OpenAI API 호출 래퍼

입력: final\_prompt  
 출력:

`{`  
  `"image_url": "...",`  
  `"refined_content": "a cute girl waving in a bright park"`  
`}`

---

## **🔟 `service/translator.py`**

revised\_prompt를 영어 → 한국어로 번역

Timeout 시 원문 유지

---

## **1️⃣1️⃣ `service/request_trace.py`**

각 단계에 request\_id 기반 추적 로그 기록

---

## **1️⃣2️⃣ `model/forbidden_to_clean.csv`**

비속어 → 순화어 사전  
 KoBART 학습 및 Rule 기반 필터에 사용

---

# **🚀 서버 실행**

`uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`

---

# **🔌 API 사용 예**

`curl -X POST http://localhost:8000/api/v1/image/generate \`  
`-H "Content-Type: application/json" \`  
`-d '{`  
  `"access_id":"u1",`  
  `"original_content":"아이가 환하게 웃고 있다",`  
  `"is_slang": false`  
`}'`

---

# **🎯 이 서버의 정체성**

이 서버는 단순한 “이미지 생성기”가 아니라:

**Korean → Safe → Webtoon → Image → Story AI Pipeline**

입니다.

