# Copilot RAG (Retrieval-Augmented Generation)

GitHub Copilot을 위한 사용자 정의 문서 검색 서버입니다. Model Context Protocol(MCP)을 구현하여 VS Code에서 GitHub Copilot이 벡터 DB에 저장된 문서를 검색할 수 있게 합니다. 커서의 Docs 기능과 유사한 기능을 무료로 구현할 수 있습니다.

## 주요 기능

- **문서 검색**: GitHub Copilot이 MCP를 통해 벡터 검색 기능을 사용할 수 있게 합니다.
- **웹사이트 크롤링**: 특정 웹사이트의 문서를 자동으로 수집하고 벡터화합니다.
- **벡터 저장소**: ChromaDB를 사용하여 문서 임베딩을 효율적으로 저장하고 검색합니다.
- **Docker 지원**: 간편한 배포와 실행을 위한 Docker 컨테이너 설정이 포함되어 있습니다.

## 설치 방법

### 사전 요구사항

- Python 3.8 이상
- Docker 및 Docker Compose (선택 사항)

### 로컬 설치

1. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/copilot-rag.git
   cd copilot-rag
   ```

2. 가상 환경 생성 및 활성화
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 서버 실행
   ```bash
   python mcp_server.py
   ```

### Docker를 사용한 설치

1. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/copilot-rag.git
   cd copilot-rag
   ```

2. Docker Compose로 빌드 및 실행
   ```bash
   docker-compose up -d
   ```

## 사용 방법

### 1. 웹사이트 크롤링

서버가 실행된 후, 다음 API를 통해 웹사이트를 크롤링하여 문서를 수집할 수 있습니다:

```bash
curl -X POST http://localhost:8765/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "site_name": "Example Docs", "max_pages": 50}'
```

### 2. VS Code 설정

1. VS Code에서 GitHub Copilot 설치 및 로그인
2. VS Code 설정(settings.json)에 다음 구성 추가:
   ```json
   "github.copilot.advanced": {
     "model.contextProvider": {
       "endpoints": [
         {
           "name": "My Documents",
           "endpoint": "http://localhost:8765/api/mcp/search"
         }
       ]
     }
   }
   ```

### 3. 문서 검색 사용

설정이 완료되면 GitHub Copilot이 자동으로 여러분의 문서를 검색하여 더 정확한 제안과 답변을 제공합니다. `/docs` 명령어로 직접 문서를 검색할 수도 있습니다.

## API 엔드포인트

- **`/api/mcp/search`**: MCP 호환 문서 검색 엔드포인트
- **`/api/crawl`**: 웹사이트 크롤링 엔드포인트
- **`/api/info`**: 서버 정보 및 통계 조회
- **`/api/reset`**: 저장된 문서 초기화
- **`/health`**: 서버 상태 확인

## 프로젝트 구조

```
copilot-rag/
├── docker-compose.yml    # Docker 설정
├── Dockerfile            # Docker 이미지 정의
├── mcp_server.py         # MCP 서버 구현
├── reference_utils.py    # 문서 처리 유틸리티
├── requirements.txt      # 의존성 목록
├── website_crawler.py    # 웹사이트 크롤링 도구
└── website_crawler_guide.md  # 크롤링 가이드
```

## 향후 계획

- 웹 인터페이스: 간단한 관리 페이지 구현 (문서 관리, 크롤링 상태 확인 등)
- 자동 업데이트: 주기적인 문서 업데이트 기능
- 데이터 영속성 개선: 백업 및 복원 기능
- 다양한 웹사이트 형식 지원: 크롤링 기능 개선

## 기여 방법

1. 저장소 포크
2. 새 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.