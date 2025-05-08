# Dockerfile - VS Code Docs 서버를 위한 Dockerfile
FROM python:3.10-slim

# 비루트 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# 필요한 시스템 패키지 설치 (curl은 healthcheck용)
RUN apt-get update && apt-get install -y curl --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 필요한 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY mcp_server.py .
COPY website_crawler.py .
COPY reference_utils.py .

# 데이터 및 로그 디렉토리 생성
RUN mkdir -p /app/data/chroma_db /app/data/models /app/logs && \
    chown -R appuser:appuser /app

# 비루트 사용자로 전환
USER appuser

# 서버 시작
CMD ["python", "mcp_server.py"]