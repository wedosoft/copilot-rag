# Dockerfile - VS Code Docs 서버를 위한 Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY mcp_server.py .
COPY website_crawler.py .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/chroma_db
RUN mkdir -p /app/data/models

# 서버 시작
CMD ["python", "mcp_server.py"]