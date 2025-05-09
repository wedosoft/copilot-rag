"""
참조 코드 모듈 - project-a에서 핵심 코드 참조

이 파일은 project-a의 핵심 코드를 참조하여 
copilot-rag 프로젝트에서 사용할 수 있게 합니다.
project-a에서 임베딩 및 벡터 검색 관련 기능을 가져왔습니다.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 임베딩을 사용하려면 주석 해제
"""
import tiktoken
import openai
from chromadb.utils import embedding_functions

# 환경 변수에서 API 키 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    # OpenAI 임베딩 함수 설정
    MODEL_NAME = "text-embedding-ada-002"
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=MODEL_NAME
    )
    
    # 토큰 인코더 초기화
    tokenizer = tiktoken.encoding_for_model(MODEL_NAME)
"""

# 기본 설정
MAX_TOKENS_PER_CHUNK = 4000  # 최대 청크 크기
CHUNK_OVERLAP = 200  # 청크 간 중복 토큰 수

def split_text_into_chunks(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    긴 텍스트를 적절한 크기의 청크로 분할합니다.
    이 함수는 project-a의 split_into_chunks 함수를 참조했습니다.
    
    Args:
        text: 분할할 텍스트
        max_tokens: 청크당 최대 토큰 수
        overlap: 청크 간 중복 토큰 수
        
    Returns:
        분할된 텍스트 청크 리스트
    """
    # 간단한 구현: 문자 기반 분할 (토큰 대신)
    avg_char_per_token = 4  # 영어 기준 평균 값
    max_chars = max_tokens * avg_char_per_token
    overlap_chars = overlap * avg_char_per_token
    
    # 텍스트가 충분히 짧으면 그대로 반환
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start_pos = 0
    
    while start_pos < len(text):
        end_pos = min(start_pos + max_chars, len(text))
        
        # 문장 또는 단락 경계에서 분할하려고 시도
        if end_pos < len(text):
            # 단락 경계에서 분할 시도
            para_end = text.rfind("\n\n", start_pos, end_pos)
            if para_end != -1 and para_end > start_pos + max_chars // 2:
                end_pos = para_end + 2
            else:
                # 문장 경계에서 분할 시도
                sentence_end = text.rfind(". ", start_pos, end_pos)
                if sentence_end != -1 and sentence_end > start_pos + max_chars // 2:
                    end_pos = sentence_end + 2
        
        chunk = text[start_pos:end_pos]
        chunks.append(chunk)
        
        # 중복을 고려한 다음 시작 위치 계산
        start_pos = end_pos - overlap_chars
        if start_pos >= len(text):
            break
    
    logger.info(f"텍스트({len(text)} 문자)를 {len(chunks)}개 청크로 분할")
    return chunks

def process_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    문서 리스트를 처리하여 필요한 경우 청크로 분할합니다.
    project-a의 process_documents 함수를 참조했습니다.
    
    Args:
        docs: 처리할 문서 리스트 (각 문서는 id, text, metadata 포함)
        
    Returns:
        처리된 문서 리스트
    """
    processed_docs = []
    
    for i, doc in enumerate(docs):
        doc_id = doc.get('id', f"doc_{i}")
        doc_text = doc.get('text', '')
        doc_metadata = doc.get('metadata', {})
        
        # 문서 길이가 기준보다 짧으면 그대로 사용
        if len(doc_text) <= MAX_TOKENS_PER_CHUNK * 4:  # 문자 기준으로 대략적 변환
            processed_docs.append(doc)
        else:
            # 긴 문서는 청크로 분할
            chunks = split_text_into_chunks(doc_text)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # 메타데이터 복사 및 청크 정보 추가
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'original_id': doc_id,
                    'is_chunk': True
                })
                
                chunk_doc = {
                    'id': f"{doc_id}_chunk_{chunk_idx}",
                    'text': chunk_text,
                    'metadata': chunk_metadata
                }
                processed_docs.append(chunk_doc)
    
    return processed_docs

def save_to_chromadb(content_data, site_name=None):
    """
    content_data(dict)를 ChromaDB에 저장하는 더미 함수(실제 구현 필요).
    Docker 환경에서 ImportError 방지용 기본 함수입니다.
    """
    logger.info(f"[더미] ChromaDB에 {len(content_data)}개 문서를 저장합니다. (site_name={site_name})")
    # 실제 저장 로직은 mcp_server.py에서 처리됨
    return True