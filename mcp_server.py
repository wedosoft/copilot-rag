"""
VS Code용 MCP 서버

이 서버는 Model Context Protocol(MCP)을 구현하여 GitHub Copilot이 벡터 DB에 저장된
문서를 검색할 수 있게 합니다. 커서의 Docs 기능과 유사한 기능을 제공합니다.
"""
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# project-a 참조 유틸리티 임포트
try:
    from reference_utils import process_documents, split_text_into_chunks
    logging.info("reference_utils 모듈을 성공적으로 로드했습니다.")
except ImportError as e:
    logging.error(f"reference_utils 모듈 로드 실패: {e}")
    # Docker 환경에서 모듈이 누락되는 경우 대비 폴백 함수 구현
    def split_text_into_chunks(text, max_tokens=4000, overlap=200):
        """텍스트를 청크로 분할하는 간단한 폴백 함수"""
        avg_char_per_token = 4  # 영어 기준 평균 문자당 토큰 수
        max_chars = max_tokens * avg_char_per_token
        overlap_chars = overlap * avg_char_per_token
        
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text) and text[end-1] != '\n' and end < len(text) - 1:
                # 단락이나 문장 경계에서 분할 시도
                next_newline = text.find('\n', end - overlap_chars)
                if next_newline != -1 and next_newline < end + overlap_chars:
                    end = next_newline + 1
                else:
                    next_period = text.find('. ', end - overlap_chars)
                    if next_period != -1 and next_period < end + overlap_chars:
                        end = next_period + 2
            
            chunks.append(text[start:end])
            start = max(start + max_chars - overlap_chars, end - overlap_chars)
            
        return chunks
    
    def process_documents(docs):
        """문서를 처리하는 간단한 폴백 함수"""
        processed_docs = []
        
        for doc in docs:
            doc_text = doc.get('text', '')
            if len(doc_text) <= 4000 * 4:  # 약 4000 토큰
                processed_docs.append(doc)
            else:
                chunks = split_text_into_chunks(doc_text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc.get('metadata', {}).copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_id': doc.get('id', ''),
                        'is_chunk': True
                    })
                    
                    processed_docs.append({
                        'id': f"{doc.get('id', '')}_chunk_{i}",
                        'text': chunk,
                        'metadata': chunk_metadata
                    })
        
        return processed_docs
    
    logging.info("reference_utils 대체 함수를 생성했습니다.")

# 로깅 설정
# logs 디렉토리 생성
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "mcp_server.log"), mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_server")

# 앱 설정
APP_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DB_DIR = os.path.join(APP_DATA_DIR, "chroma_db")
MODEL_DIR = os.path.join(APP_DATA_DIR, "models")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# 디렉토리 생성
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# FastAPI 앱 생성
app = FastAPI(title="VS Code Docs MCP Server",
              description="GitHub Copilot을 위한 문서 검색 MCP 서버")

# CORS 설정 - VS Code에서 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 정의
class MCPRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class MCPSearchResult(BaseModel):
    title: str
    content: str
    url: Optional[str] = None
    score: float

class MCPResponse(BaseModel):
    results: List[MCPSearchResult]
    message: Optional[str] = None

class CrawlRequest(BaseModel):
    url: str
    site_name: str
    max_pages: Optional[int] = 50
    exclusion_patterns: Optional[List[str]] = None
    resume: Optional[bool] = False
    rate_limit_delay: Optional[float] = 0.5
    max_workers: Optional[int] = 5

class CrawlStatus(BaseModel):
    status: str
    site_name: str
    url: str
    pages_crawled: int
    in_progress: bool
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    message: Optional[str] = None

# 전역 변수
embedding_model = None
chroma_client = None
collection = None
crawl_tasks = {}  # 크롤링 작업 추적용

def initialize_embedding_model():
    """임베딩 모델 초기화"""
    global embedding_model
    if embedding_model is None:
        logger.info("임베딩 모델 초기화 중...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR)
    return embedding_model

def initialize_chroma_db():
    """ChromaDB 초기화"""
    global chroma_client, collection
    if (chroma_client is None) or (collection is None):
        logger.info("ChromaDB 초기화 중...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # 컬렉션 생성 또는 가져오기
        try:
            collection = chroma_client.get_collection("documents")
            logger.info(f"기존 컬렉션 불러옴: {collection.count()} 문서")
        except:
            collection = chroma_client.create_collection("documents")
            logger.info("새 컬렉션 생성됨")
    
    return collection

# 임베딩 함수
def embed_text(text: str) -> List[float]:
    """텍스트를 임베딩 벡터로 변환"""
    model = initialize_embedding_model()
    return model.encode(text).tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """여러 텍스트를 임베딩 벡터로 변환"""
    model = initialize_embedding_model()
    return model.encode(texts).tolist()

# 백그라운드 크롤링 작업
async def background_crawl(url: str, site_name: str, max_pages: int = 50, 
                           exclusion_patterns: List[str] = None, resume: bool = False,
                           rate_limit_delay: float = 0.5, max_workers: int = 5):
    """웹사이트 크롤링을 백그라운드에서 실행"""
    from website_crawler import WebsiteCrawler
    
    # 임시 출력 디렉토리
    temp_dir = os.path.join(APP_DATA_DIR, "temp_crawl", site_name)
    os.makedirs(temp_dir, exist_ok=True)
    
    crawl_tasks[site_name] = {
        "status": "starting",
        "site_name": site_name,
        "url": url,
        "pages_crawled": 0,
        "in_progress": True,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "message": "크롤링 시작 중..."
    }
    
    try:
        # 웹사이트 크롤러 초기화
        crawler = WebsiteCrawler(
            start_url=url,
            output_dir=temp_dir,
            site_name=site_name,
            max_pages=max_pages,
            resume=resume
        )
        
        # 설정 적용
        crawler.rate_limit_delay = rate_limit_delay
        if exclusion_patterns:
            for pattern in exclusion_patterns:
                crawler.add_exclusion_pattern(pattern)
        
        # 크롤링 상태 업데이트
        crawl_tasks[site_name]["status"] = "crawling"
        crawl_tasks[site_name]["message"] = "웹사이트 크롤링 중..."
        
        # 크롤링 실행
        documents = crawler.crawl(max_pages=max_pages, max_workers=max_workers)
        
        # 크롤링 결과 확인
        if not documents or len(documents) == 0:
            crawl_tasks[site_name]["status"] = "failed"
            crawl_tasks[site_name]["message"] = "크롤링 결과가 없습니다."
            crawl_tasks[site_name]["in_progress"] = False
            crawl_tasks[site_name]["end_time"] = datetime.now().isoformat()
            return
        
        # 크롤링 상태 업데이트
        crawl_tasks[site_name]["pages_crawled"] = len(documents)
        crawl_tasks[site_name]["status"] = "processing"
        crawl_tasks[site_name]["message"] = "크롤링된 문서 처리 중..."
        
        # ChromaDB 초기화
        collection = initialize_chroma_db()
        
        # 문서 형식 변환 (project-a 참조 유틸리티 사용)
        processed_doc_list = []
        for idx, doc_data in documents.items():
            doc_id = f"{site_name}_{idx}"
            processed_doc_list.append({
                'id': doc_id,
                'text': doc_data['full_text'],
                'metadata': {
                    'title': doc_data['title'],
                    'url': doc_data['url'],
                    'site': site_name,
                    'created_at': datetime.now().isoformat()
                }
            })
        
        # project-a 참조 유틸리티로 문서 처리 (긴 문서 청크 분할)
        processed_docs = process_documents(processed_doc_list)
        
        # 크롤링 상태 업데이트
        crawl_tasks[site_name]["status"] = "storing"
        crawl_tasks[site_name]["message"] = "벡터 DB에 문서 저장 중..."
        
        texts = []
        ids = []
        metadatas = []
        
        for doc in processed_docs:
            # 이미 존재하는 문서인지 확인 (사이트 이름과 URL 기준)
            try:
                # ChromaDB는 where 조건에 하나의 연산자만 허용하므로 $and 연산자 사용
                existing = collection.get(
                    where={"$and": [
                        {"site": {"$eq": site_name}},
                        {"url": {"$eq": doc['metadata'].get('url', '')}}
                    ]}
                )
                if existing and len(existing["ids"]) > 0:
                    # 이미 존재하는 문서 삭제
                    collection.delete(ids=existing["ids"])
            except Exception as e:
                logger.error(f"문서 검사 중 오류: {str(e)}")
            
            texts.append(doc['text'])
            ids.append(doc['id'])
            metadatas.append(doc['metadata'])
        
        # 임베딩 생성 및 ChromaDB에 추가
        if texts:
            embeddings = embed_texts(texts)
            
            # 청크 단위로 추가 (메모리 이슈 방지)
            chunk_size = 50
            for i in range(0, len(texts), chunk_size):
                end = min(i + chunk_size, len(texts))
                collection.add(
                    documents=texts[i:end],
                    embeddings=embeddings[i:end],
                    ids=ids[i:end],
                    metadatas=metadatas[i:end]
                )
                
                # 진행 상황 업데이트
                progress = min(100, int((end / len(texts)) * 100))
                crawl_tasks[site_name]["message"] = f"벡터 DB에 문서 저장 중... ({progress}%)"
        
        # 완료 상태 업데이트
        crawl_tasks[site_name]["status"] = "completed"
        crawl_tasks[site_name]["message"] = "크롤링 및 문서 저장 완료"
        crawl_tasks[site_name]["in_progress"] = False
        crawl_tasks[site_name]["end_time"] = datetime.now().isoformat()
        logger.info(f"사이트 '{site_name}'의 {len(texts)}개 문서 크롤링 및 저장 완료")
        
    except Exception as e:
        logger.error(f"크롤링 오류: {str(e)}")
        crawl_tasks[site_name]["status"] = "failed"
        crawl_tasks[site_name]["message"] = f"크롤링 오류: {str(e)}"
        crawl_tasks[site_name]["in_progress"] = False
        crawl_tasks[site_name]["end_time"] = datetime.now().isoformat()

# MCP 호환 검색 API 엔드포인트
@app.post("/api/mcp/search", response_model=MCPResponse)
async def mcp_search(request: MCPRequest):
    try:
        # ChromaDB 초기화
        collection = initialize_chroma_db()
        
        if collection.count() == 0:
            return MCPResponse(
                results=[],
                message="문서가 없습니다. 먼저 /api/crawl 엔드포인트로 웹사이트를 크롤링하세요."
            )
        
        # 쿼리 임베딩
        query_embedding = embed_text(request.query)
        
        # ChromaDB 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.max_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # 결과 포맷팅
        formatted_results = []
        
        if results["documents"] and len(results["documents"][0]) > 0:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # 유사도 점수 계산 (거리를 유사도로 변환)
                similarity = 1.0 - min(distance, 1.0)
                
                title = metadata.get("title", "문서")
                url = metadata.get("url", None)
                
                # 내용 제한 (너무 길면 잘라냄)
                content = doc[:800] + "..." if len(doc) > 800 else doc
                
                formatted_results.append(MCPSearchResult(
                    title=title,
                    content=content,
                    url=url,
                    score=float(similarity)
                ))
        
        return MCPResponse(results=formatted_results)
    
    except Exception as e:
        logger.error(f"검색 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")

# MCP 메타데이터 엔드포인트
@app.get("/api/mcp/metadata")
async def mcp_metadata():
    return {
        "name": "VS Code Docs MCP Server",
        "version": "1.0.0",
        "description": "Vector document search using Model Context Protocol",
        "endpoints": {
            "search": "/api/mcp/search"
        },
        "capabilities": ["document_search", "vector_similarity"]
    }

# 웹사이트 크롤링 엔드포인트 (비동기 처리)
@app.post("/api/crawl")
async def crawl_website(request: CrawlRequest, background_tasks: BackgroundTasks):
    try:
        site_name = request.site_name
        
        # 입력 유효성 검증
        if not site_name:
            raise HTTPException(status_code=400, detail="'site_name' 필드가 필요합니다.")
        if not request.url:
            raise HTTPException(status_code=400, detail="'url' 필드가 필요합니다.")
        
        # 이미 진행 중인 크롤링 작업이 있는지 확인
        if site_name in crawl_tasks and crawl_tasks[site_name].get("in_progress", False):
            return {
                "status": "already_running",
                "message": f"'{site_name}' 사이트의 크롤링이 이미 진행 중입니다.",
                "task_info": crawl_tasks[site_name]
            }
        
        # 기본 제외 패턴
        exclusion_patterns = request.exclusion_patterns or [
            r'/wp-admin/',
            r'/login/',
            r'/search/',
            r'/cart/',
            r'/account/',
            r'\?s=',  # 검색 쿼리
            r'\?p=\d+&',  # 페이지네이션 쿼리
            r'/page/\d+/$'  # 페이지네이션 URL
        ]
        
        # 크롤링 작업 정보 초기화 (응답 전에 상태를 설정)
        crawl_tasks[site_name] = {
            "status": "starting",
            "site_name": site_name,
            "url": request.url,
            "pages_crawled": 0,
            "in_progress": True,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "message": "크롤링 작업이 대기열에 추가됨"
        }
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            background_crawl,
            url=request.url,
            site_name=site_name,
            max_pages=request.max_pages,
            exclusion_patterns=exclusion_patterns,
            resume=request.resume,
            rate_limit_delay=request.rate_limit_delay,
            max_workers=request.max_workers
        )
        
        # 작업이 시작되었음을 알리는 응답
        return {
            "status": "started",
            "message": f"'{site_name}' 사이트 크롤링 작업이 시작되었습니다. /api/crawl/status/{site_name}에서 진행 상황을 확인하세요.",
            "site_name": site_name,
            "max_pages": request.max_pages
        }
    except Exception as e:
        logger.error(f"크롤링 시작 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"크롤링 시작 오류: {str(e)}")

# 크롤링 상태 확인 엔드포인트
@app.get("/api/crawl/status/{site_name}", response_model=CrawlStatus)
async def get_crawl_status(site_name: str):
    if site_name not in crawl_tasks:
        raise HTTPException(status_code=404, detail=f"'{site_name}' 사이트의 크롤링 작업을 찾을 수 없습니다.")
    
    task_info = crawl_tasks[site_name]
    return CrawlStatus(**task_info)

# 크롤링 작업 목록 조회
@app.get("/api/crawl/tasks")
async def list_crawl_tasks():
    return {
        "tasks": list(crawl_tasks.keys()),
        "details": crawl_tasks
    }

# 컬렉션 정보 엔드포인트
@app.get("/api/info")
async def get_info():
    try:
        collection = initialize_chroma_db()
        count = collection.count()
        
        # 컬렉션 내 사이트 목록 가져오기
        sites = {}
        if count > 0:
            try:
                # 컬렉션에서 모든 메타데이터 가져오기
                results = collection.get(include=["metadatas"])
                
                if results and "metadatas" in results:
                    for metadata in results["metadatas"]:
                        site = metadata.get("site", "unknown")
                        if site in sites:
                            sites[site] += 1
                        else:
                            sites[site] = 1
            except Exception as e:
                logger.error(f"메타데이터 가져오는 중 오류: {str(e)}")
        
        return {
            "status": "healthy",
            "document_count": count,
            "sites": sites,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정보 조회 오류: {str(e)}")

# 사이트별 문서 삭제 엔드포인트
@app.delete("/api/sites/{site_name}")
async def delete_site_documents(site_name: str):
    try:
        collection = initialize_chroma_db()
        
        # 사이트 이름으로 문서 검색
        results = collection.get(
            where={"site": {"$eq": site_name}},
            include=["metadatas", "documents", "embeddings", "ids"]
        )
        
        if not results or "ids" not in results or not results["ids"]:
            return {
                "status": "not_found",
                "message": f"'{site_name}' 사이트의 문서를 찾을 수 없습니다."
            }
        
        # 문서 ID 가져오기
        doc_ids = results["ids"]
        
        # 문서 삭제
        collection.delete(ids=doc_ids)
        
        return {
            "status": "success",
            "message": f"'{site_name}' 사이트의 {len(doc_ids)}개 문서가 삭제되었습니다.",
            "document_count": collection.count()
        }
    except Exception as e:
        logger.error(f"사이트 문서 삭제 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"문서 삭제 오류: {str(e)}")

# 컬렉션 삭제 엔드포인트
@app.delete("/api/reset")
async def reset_collection():
    try:
        global chroma_client, collection
        
        # ChromaDB 클라이언트 초기화
        initialize_chroma_db()
        
        # 컬렉션 삭제 및 재생성
        try:
            chroma_client.delete_collection("documents")
        except:
            pass
        
        collection = chroma_client.create_collection("documents")
        
        return {
            "status": "success",
            "message": "모든 문서가 삭제되었습니다.",
            "document_count": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"컬렉션 초기화 오류: {str(e)}")

# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 앱 초기화
@app.on_event("startup")
async def startup_event():
    logger.info("MCP 서버 시작 중...")
    initialize_embedding_model()
    initialize_chroma_db()
    logger.info("MCP 서버 초기화 완료")

# 서버 실행
if __name__ == "__main__":
    logger.info("VS Code Docs MCP 서버 시작 중...")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8765, reload=True)