"""
VS Code용 MCP 서버

이 서버는 Model Context Protocol(MCP)을 구현하여 GitHub Copilot이 벡터 DB에 저장된
문서를 검색할 수 있게 합니다. 커서의 Docs 기능과 유사한 기능을 제공합니다.
"""
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# project-a 참조 유틸리티 임포트
from reference_utils import process_documents, split_text_into_chunks

# 앱 설정
APP_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DB_DIR = os.path.join(APP_DATA_DIR, "chroma_db")
MODEL_DIR = os.path.join(APP_DATA_DIR, "models")

# 디렉토리 생성
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

# 전역 변수
embedding_model = None
chroma_client = None
collection = None

def initialize_embedding_model():
    """임베딩 모델 초기화"""
    global embedding_model
    if embedding_model is None:
        print("임베딩 모델 초기화 중...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR)
    return embedding_model

def initialize_chroma_db():
    """ChromaDB 초기화"""
    global chroma_client, collection
    if chroma_client is None:
        print("ChromaDB 초기화 중...")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # 컬렉션 생성 또는 가져오기
        try:
            collection = chroma_client.get_collection("documents")
            print(f"기존 컬렉션 불러옴: {collection.count()} 문서")
        except:
            collection = chroma_client.create_collection("documents")
            print("새 컬렉션 생성됨")
    
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
        print(f"검색 오류: {str(e)}")
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

# 웹사이트 크롤링 엔드포인트
@app.post("/api/crawl")
async def crawl_website(request: CrawlRequest):
    try:
        # 웹사이트 크롤러 가져오기
        from website_crawler import WebsiteCrawler
        
        # 임시 출력 디렉토리
        temp_dir = os.path.join(APP_DATA_DIR, "temp_crawl")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 웹사이트 크롤링
        crawler = WebsiteCrawler(
            start_url=request.url,
            output_dir=temp_dir,
            site_name=request.site_name
        )
        documents = crawler.crawl(max_pages=request.max_pages)
        
        # ChromaDB 초기화
        collection = initialize_chroma_db()
        
        # 크롤링된 문서 처리 및 ChromaDB에 추가
        print(f"{len(documents)} 문서를 처리 중...")
        
        # 문서 형식 변환 (project-a 참조 유틸리티 사용)
        processed_doc_list = []
        for idx, doc_data in documents.items():
            doc_id = f"{request.site_name}_{idx}"
            processed_doc_list.append({
                'id': doc_id,
                'text': doc_data['full_text'],
                'metadata': {
                    'title': doc_data['title'],
                    'url': doc_data['url'],
                    'site': request.site_name,
                    'created_at': datetime.now().isoformat()
                }
            })
        
        # project-a 참조 유틸리티로 문서 처리 (긴 문서 청크 분할)
        processed_docs = process_documents(processed_doc_list)
        
        texts = []
        ids = []
        metadatas = []
        
        for doc in processed_docs:
            # 이미 존재하는 문서인지 확인 (사이트 이름과 URL 기준)
            try:
                existing = collection.get(
                    where={"site": request.site_name, "url": doc['metadata'].get('url', '')}
                )
                if existing and len(existing["ids"]) > 0:
                    # 이미 존재하는 문서 삭제
                    collection.delete(ids=existing["ids"])
            except Exception as e:
                print(f"문서 검사 중 오류: {str(e)}")
            
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
        
        return {
            "status": "success",
            "message": f"{len(texts)} 문서가 성공적으로 크롤링되고 저장되었습니다.",
            "document_count": collection.count()
        }
    
    except Exception as e:
        print(f"크롤링 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"크롤링 오류: {str(e)}")

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
                print(f"메타데이터 가져오는 중 오류: {str(e)}")
        
        return {
            "status": "healthy",
            "document_count": count,
            "sites": sites,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정보 조회 오류: {str(e)}")

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
    initialize_embedding_model()
    initialize_chroma_db()

# 서버 실행
if __name__ == "__main__":
    print("VS Code Docs MCP 서버 시작 중...")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8765, reload=True)