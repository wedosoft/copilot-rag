import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time
import json
import argparse
import configparser
from pathlib import Path
import shutil
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import pickle
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebsiteCrawler")

# 중앙 저장소 경로 설정 (iCloud Drive 내 폴더로 변경)
HOME_DIR = Path.home()
ICLOUD_DIR = os.path.join(HOME_DIR, "Library/Mobile Documents/com~apple~CloudDocs")  # macOS의 iCloud Drive 경로
DOCS_BASE_DIR = os.path.join(ICLOUD_DIR, "DocsSearch")  # iCloud Drive 내의 DocsSearch 폴더
CONFIG_FILE = os.path.join(DOCS_BASE_DIR, "config.ini")

# 기존 경로(로컬)도 유지 (폴백용)
LOCAL_DOCS_DIR = os.path.join(HOME_DIR, ".docsearch")

# 저장소 위치 선택 함수 추가
def get_storage_dir():
    """저장소 위치 선택 (iCloud 우선, 없으면 로컬)"""
    if os.path.exists(ICLOUD_DIR):
        return DOCS_BASE_DIR
    else:
        print("iCloud Drive를 찾을 수 없습니다. 로컬 저장소를 사용합니다.")
        return LOCAL_DOCS_DIR

# 실제 사용할 저장소 경로 설정
DOCS_BASE_DIR = get_storage_dir()
CONFIG_FILE = os.path.join(DOCS_BASE_DIR, "config.ini")

class WebsiteCrawler:
    def __init__(self, start_url, output_dir="crawled_content", site_name=None, max_pages=5000, resume=False):
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.visited_urls = set()
        self.content_data = {}
        self.output_dir = output_dir
        self.site_name = site_name or self.base_domain
        self.max_pages = max_pages
        self.resume = resume
        self.rate_limit_delay = 0.5  # 서버 부하 방지를 위한 기본 딜레이 (초)
        self.retry_count = 3  # 요청 실패 시 재시도 횟수
        self.url_exclusion_patterns = []  # 제외할 URL 패턴 (정규식)
        
        # 출력 디렉토리 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 크롤링 상태 파일 경로
        self.state_file = os.path.join(output_dir, 'crawler_state.pkl')
        self.resume_file = os.path.join(output_dir, 'crawled_data.json')
        
        # 이전 상태 복원 (resume=True인 경우)
        if resume and self._load_state():
            logger.info(f"이전 크롤링 상태를 불러왔습니다. 이미 방문한 URL: {len(self.visited_urls)}개")
            # 이전에 크롤링한 데이터 불러오기
            if os.path.exists(self.resume_file):
                try:
                    with open(self.resume_file, 'r', encoding='utf-8') as f:
                        self.content_data = json.load(f)
                    logger.info(f"이전에 크롤링한 {len(self.content_data)}개의 페이지 데이터를 불러왔습니다.")
                except Exception as e:
                    logger.error(f"이전 데이터를 불러오는 중 오류 발생: {e}")

    def add_exclusion_pattern(self, pattern):
        """크롤링에서 제외할 URL 패턴 추가 (정규식)"""
        self.url_exclusion_patterns.append(re.compile(pattern))
        
    def should_exclude_url(self, url):
        """URL이 제외 패턴에 해당하는지 확인"""
        for pattern in self.url_exclusion_patterns:
            if pattern.search(url):
                return True
        return False

    def _save_state(self):
        """크롤링 상태 저장"""
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump({'visited_urls': self.visited_urls}, f)
            return True
        except Exception as e:
            logger.error(f"크롤링 상태 저장 중 오류 발생: {e}")
            return False
            
    def _load_state(self):
        """크롤링 상태 불러오기"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.visited_urls = state.get('visited_urls', set())
                return True
            except Exception as e:
                logger.error(f"크롤링 상태 불러오기 중 오류 발생: {e}")
        return False

    def is_valid_url(self, url):
        """URL이 유효한지, 같은 도메인에 속하는지 확인"""
        parsed = urlparse(url)
        
        # 기본 유효성 검사
        if not (bool(parsed.netloc) and parsed.netloc == self.base_domain):
            return False
            
        # 제외 패턴 확인
        if self.should_exclude_url(url):
            logger.info(f"제외 패턴에 의해 URL 제외: {url}")
            return False
            
        # 일반적으로 크롤링이 불필요한 파일 형식 제외
        excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js', '.ico']
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
            
        return True
    
    def get_page_content(self, url):
        """페이지 내용 가져오기 (재시도 로직 추가)"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(self.retry_count):
            try:
                logger.info(f"페이지 요청 중: {url} (시도 {attempt+1}/{self.retry_count})")
                response = requests.get(url, timeout=30, headers=headers)
                
                # 429 (Too Many Requests) 응답 처리
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', self.rate_limit_delay * 4))
                    logger.warning(f"Rate limit 발생, {wait_time}초 대기 후 재시도합니다: {url}")
                    time.sleep(wait_time)
                    continue
                    
                # 다른 오류 응답 처리
                if response.status_code != 200:
                    logger.warning(f"HTTP 오류 {response.status_code}: {url}, {self.rate_limit_delay * 2}초 대기 후 재시도")
                    time.sleep(self.rate_limit_delay * 2)
                    continue
                
                # 성공 응답
                return response.text
                
            except requests.RequestException as e:
                logger.error(f"요청 오류: {url}, {e}, {self.rate_limit_delay * 2}초 대기 후 재시도")
                time.sleep(self.rate_limit_delay * 2)
        
        # 모든 시도 실패
        logger.error(f"최대 재시도 횟수 초과, 페이지 건너뜀: {url}")
        return None
    
    def extract_links(self, url, html_content):
        """페이지에서 링크 추출 (더 효율적인 방법으로 개선)"""
        if not html_content:
            return []
            
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        # 혹시 모를 무한 루프 방지를 위해 최대 링크 수 제한
        max_links_per_page = 500
        link_count = 0
        
        for a_tag in soup.find_all('a', href=True):
            if link_count >= max_links_per_page:
                logger.warning(f"최대 링크 수 ({max_links_per_page})에 도달하여 링크 추출 중단: {url}")
                break
                
            href = a_tag['href']
            absolute_url = urljoin(url, href)
            
            # 같은 도메인이고 아직 방문하지 않은 URL만 추가 (is_valid_url이 제외 패턴 체크도 포함)
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
                link_count += 1
        
        return links
    
    def extract_content(self, url, html_content):
        """페이지 내용 추출"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 제목 추출
        title = soup.title.string if soup.title else "No Title"
        title = title.strip()
        
        # 메인 콘텐츠 영역 찾기 시도
        main_content = None
        for selector in ['main', 'article', '.content', '#content', '.post', '.entry', '.post-content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # 메인 콘텐츠를 찾지 못한 경우 body 사용
        if not main_content:
            main_content = soup.body
        
        # 불필요한 요소 제거
        for tag in main_content.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # 본문 내용 추출 (h1, h2, h3, h4, h5, h6, p 태그 등)
        content = []
        for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'code', 'pre']):
            text = tag.get_text().strip()
            if text:
                content.append({
                    'tag': tag.name,
                    'text': text
                })
        
        # 전체 텍스트 내용도 저장
        full_text = title + "\n\n"
        for item in content:
            if item['tag'].startswith('h'):
                full_text += f"{'#' * int(item['tag'][1:])} {item['text']}\n\n"
            else:
                full_text += f"{item['text']}\n\n"
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'full_text': full_text
        }
    
    def chunked_save(self, chunk_size=1000):
        """크롤링 데이터를 청크로 나누어 저장 (대량 데이터 처리용)"""
        if not self.content_data:
            logger.warning("저장할 데이터가 없습니다.")
            return
            
        os.makedirs(os.path.join(self.output_dir, 'chunks'), exist_ok=True)
        
        # 청크로 나누어 저장
        chunks = {}
        chunk_num = 0
        items = list(self.content_data.items())
        
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i+chunk_size])
            chunk_file = os.path.join(self.output_dir, 'chunks', f'chunk_{chunk_num}.json')
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
                
            chunks[f'chunk_{chunk_num}'] = {
                'file': chunk_file,
                'count': len(chunk),
                'start_id': list(chunk.keys())[0],
                'end_id': list(chunk.keys())[-1]
            }
            
            chunk_num += 1
        
        # 청크 정보를 인덱스 파일에 저장
        index_file = os.path.join(self.output_dir, 'chunks', 'index.json')
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_pages': len(self.content_data),
                'chunk_size': chunk_size,
                'chunk_count': chunk_num,
                'chunks': chunks,
                'crawl_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
            
        logger.info(f"데이터를 {chunk_num}개 청크로 나누어 저장했습니다. 총 {len(self.content_data)}개 페이지.")
        
        # 전체 데이터도 저장 (기존 방식)
        with open(os.path.join(self.output_dir, 'crawled_data.json'), 'w', encoding='utf-8') as f:
            json.dump(self.content_data, f, ensure_ascii=False, indent=2)
    
    def crawl(self, max_pages=None, max_workers=5, save_interval=100):
        """웹사이트 크롤링 시작 (대규모 사이트 처리 개선)"""
        # max_pages가 None이면 인스턴스 변수 사용
        if max_pages is None:
            max_pages = self.max_pages
        
        # 사이트맵 확인
        sitemap_urls = []
        try:
            logger.info("사이트맵 확인 중...")
            sitemap_urls = self.check_sitemaps()
            if sitemap_urls:
                logger.info(f"사이트맵에서 {len(sitemap_urls)}개의 URL을 발견했습니다.")
                # 사이트맵 URL을 대기열에 추가
                queue = sitemap_urls + ([self.start_url] if not self.resume else [])
            else:
                # 사이트맵이 없으면 기본 시작 URL 사용
                queue = [self.start_url] if not self.resume else [url for url in self.visited_urls if url != self.start_url]
        except Exception as e:
            logger.error(f"사이트맵 확인 중 오류 발생: {e}")
            # 오류 발생 시 기본 시작 URL 사용
            queue = [self.start_url] if not self.resume else [url for url in self.visited_urls if url != self.start_url]
            
        # 이미 크롤링된 페이지 수
        page_count = len(self.content_data)
        logger.info(f"크롤링 시작: 목표 {max_pages}페이지, 이미 {page_count}페이지 완료")
        
        # 시간 측정 시작
        start_time = time.time()
        last_save_time = start_time
        
        while queue and page_count < max_pages:
            # 중간 저장 (일정 간격으로)
            current_time = time.time()
            if current_time - last_save_time > 60:  # 1분마다 저장
                self.save_content()
                self._save_state()
                last_save_time = current_time
                logger.info(f"중간 저장 완료: {page_count}/{max_pages} 페이지 (경과 시간: {(current_time-start_time)/60:.1f}분)")
            
            # 남은 페이지 수 계산 및 로그
            remaining = max_pages - page_count
            logger.info(f"남은 페이지: {remaining}, 대기 중인 URL: {len(queue)}")
            
            # 여러 URL을 병렬로 처리
            batch_size = min(max_workers, len(queue), max_pages - page_count)
            current_batch = queue[:batch_size]
            queue = queue[batch_size:]
            
            # 결과 저장할 임시 딕셔너리
            new_results = {}
            new_links = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def process_url(url):
                    if url in self.visited_urls and not self.resume:
                        return None, []
                    
                    logger.info(f"Crawling: {url}")
                    self.visited_urls.add(url)
                    
                    html_content = self.get_page_content(url)
                    if not html_content:
                        return None, []
                    
                    # 내용 추출
                    page_data = self.extract_content(url, html_content)
                    
                    # 링크 추출
                    links = self.extract_links(url, html_content)
                    
                    # 서버 부하 방지를 위한 딜레이
                    time.sleep(self.rate_limit_delay)
                    
                    return page_data, links
                
                # 병렬 실행
                futures = {executor.submit(process_url, url): url for url in current_batch}
                for future in futures:
                    url = futures[future]
                    try:
                        page_data, links = future.result()
                        if page_data:
                            new_results[str(page_count)] = page_data
                            page_count += 1
                            new_links.extend(links)
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
            
            # 새 결과 및 링크 업데이트
            self.content_data.update(new_results)
            queue.extend([link for link in new_links if link not in self.visited_urls])
            
            # 일정 간격으로 중간 저장
            if page_count % save_interval == 0:
                self.save_content()
                self._save_state()
        
        # 최종 저장 (청크 방식과 일반 방식 모두)
        self.save_content()
        self.chunked_save()
        self._save_state()
        
        # 크롤링 통계
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"크롤링 완료: {page_count}페이지, 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        
        return self.content_data
    
    def check_sitemaps(self):
        """웹사이트의 사이트맵이 있는지 확인하고 URL 추출"""
        sitemap_urls = []
        
        # 일반적인 사이트맵 위치 확인
        sitemap_locations = [
            'sitemap.xml',
            'sitemap_index.xml',
            'sitemap/sitemap.xml',
            'sitemaps/sitemap.xml'
        ]
        
        # robots.txt에서 사이트맵 URL 확인
        try:
            robots_url = urljoin(self.start_url, '/robots.txt')
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        sitemap_locations.append(sitemap_url)
        except Exception as e:
            logger.warning(f"robots.txt 확인 중 오류 발생: {e}")
            
        # 각 사이트맵 위치 확인
        for loc in sitemap_locations:
            try:
                sitemap_url = urljoin(self.start_url, loc)
                response = requests.get(sitemap_url, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"사이트맵 발견: {sitemap_url}")
                    
                    # XML 파싱
                    soup = BeautifulSoup(response.text, 'xml')
                    
                    # sitemap 인덱스 파일 처리
                    for sitemap in soup.find_all('sitemap'):
                        loc_tag = sitemap.find('loc')
                        if loc_tag:
                            sub_sitemap_url = loc_tag.text
                            try:
                                sub_response = requests.get(sub_sitemap_url, timeout=10)
                                if sub_response.status_code == 200:
                                    sub_soup = BeautifulSoup(sub_response.text, 'xml')
                                    for url in sub_soup.find_all('url'):
                                        loc_tag = url.find('loc')
                                        if loc_tag and self.is_valid_url(loc_tag.text):
                                            sitemap_urls.append(loc_tag.text)
                            except Exception as e:
                                logger.warning(f"하위 사이트맵 처리 중 오류 발생: {e}")
                    
                    # 일반 sitemap 파일 처리
                    for url in soup.find_all('url'):
                        loc_tag = url.find('loc')
                        if loc_tag and self.is_valid_url(loc_tag.text):
                            sitemap_urls.append(loc_tag.text)
                            
            except Exception as e:
                logger.warning(f"사이트맵 URL 처리 중 오류 발생: {sitemap_url}, {e}")
                
        # 중복 제거 및 고유 URL 목록 반환
        return list(set(sitemap_urls))
    
    def save_content(self):
        """크롤링한 내용 저장"""
        # JSON으로만 저장
        with open(os.path.join(self.output_dir, 'crawled_data.json'), 'w', encoding='utf-8') as f:
            json.dump(self.content_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"크롤링 완료: {len(self.content_data)} 페이지가 {self.output_dir}에 저장되었습니다.")


# 설정 관리 클래스
class DocsManager:
    def __init__(self):
        # 기본 디렉토리 생성
        if not os.path.exists(DOCS_BASE_DIR):
            os.makedirs(DOCS_BASE_DIR)
        
        # 설정 파일 로드 또는 생성
        self.config = configparser.ConfigParser()
        if os.path.exists(CONFIG_FILE):
            self.config.read(CONFIG_FILE)
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """기본 설정 파일 생성"""
        self.config["GENERAL"] = {
            "default_max_pages": "50"
        }
        self.config["SITES"] = {}
        self._save_config()
    
    def _save_config(self):
        """설정 파일 저장"""
        with open(CONFIG_FILE, 'w') as f:
            self.config.write(f)
    
    def list_sites(self):
        """크롤링된 사이트 목록 표시"""
        sites = self.config["SITES"]
        if not sites:
            print("등록된 사이트가 없습니다.")
            return
        
        print("\n등록된 사이트 목록:")
        for name, url in sites.items():
            site_dir = os.path.join(DOCS_BASE_DIR, name)
            if os.path.exists(site_dir):
                size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                          for dirpath, _, filenames in os.walk(site_dir) 
                          for filename in filenames)
                size_mb = size / (1024 * 1024)
                print(f"- {name}: {url} (크기: {size_mb:.2f} MB)")
            else:
                print(f"- {name}: {url} (데이터 없음)")
    
    def add_site(self, name, url, max_pages=5000, resume=False, exclusion_patterns=None, 
                 rate_limit_delay=0.5, chunk_size=1000, max_workers=5):
        """새 사이트 추가 및 크롤링"""
        # 이름 검증
        if name in self.config["SITES"] and not resume:
            print(f"'{name}' 사이트가 이미 존재합니다. 다른 이름을 사용하세요.")
            return False
        
        # 사이트 디렉토리 경로
        site_dir = os.path.join(DOCS_BASE_DIR, name)
        
        # 크롤링 실행
        try:
            crawler = WebsiteCrawler(url, output_dir=site_dir, site_name=name, max_pages=max_pages, resume=resume)
            
            # 크롤링 설정 적용
            crawler.rate_limit_delay = rate_limit_delay
            
            # 제외 패턴 추가
            if exclusion_patterns:
                for pattern in exclusion_patterns:
                    crawler.add_exclusion_pattern(pattern)
            
            # 크롤링 실행
            crawler.crawl(max_pages=max_pages, max_workers=max_workers, save_interval=chunk_size)
            
            # 대용량 데이터를 청크로 나누어 저장
            crawler.chunked_save(chunk_size=chunk_size)
            
            # 설정 파일에 사이트 추가
            self.config["SITES"][name] = url
            self._save_config()
            
            print(f"'{name}' 사이트가 성공적으로 크롤링되었습니다.")
            print(f"데이터는 {site_dir} 디렉토리에 저장되었습니다.")
            return True
        except Exception as e:
            print(f"크롤링 중 오류 발생: {e}")
            # resume 모드에서는 디렉토리를 삭제하지 않음
            if not resume and os.path.exists(site_dir):
                shutil.rmtree(site_dir)
            return False
    
    def remove_site(self, name):
        """사이트 제거"""
        if name not in self.config["SITES"]:
            print(f"'{name}' 사이트가 존재하지 않습니다.")
            return False
        
        # 설정에서 제거
        self.config["SITES"].pop(name)
        self._save_config()
        
        # 디렉토리 제거
        site_dir = os.path.join(DOCS_BASE_DIR, name)
        if os.path.exists(site_dir):
            shutil.rmtree(site_dir)
        
        print(f"'{name}' 사이트가 성공적으로 제거되었습니다.")
        return True
    
    def update_site(self, name, max_pages=5000, resume=False, exclusion_patterns=None, 
                    rate_limit_delay=0.5, chunk_size=1000, max_workers=5):
        """기존 사이트 업데이트"""
        if name not in self.config["SITES"]:
            print(f"'{name}' 사이트가 존재하지 않습니다.")
            return False
        
        url = self.config["SITES"][name]
        site_dir = os.path.join(DOCS_BASE_DIR, name)
        
        # 이어서 진행이 아닌 경우에만 백업
        if not resume:
            # 기존 데이터 백업
            backup_dir = f"{site_dir}_backup_{int(time.time())}"
            if os.path.exists(site_dir):
                shutil.copytree(site_dir, backup_dir)
        
        # 재크롤링
        try:
            # 이어서 진행이 아닌 경우에만 디렉토리 삭제
            if not resume and os.path.exists(site_dir):
                shutil.rmtree(site_dir)
            
            # 새로 크롤링 또는 이어서 크롤링
            crawler = WebsiteCrawler(url, output_dir=site_dir, site_name=name, max_pages=max_pages, resume=resume)
            
            # 크롤링 설정 적용
            crawler.rate_limit_delay = rate_limit_delay
            
            # 제외 패턴 추가
            if exclusion_patterns:
                for pattern in exclusion_patterns:
                    crawler.add_exclusion_pattern(pattern)
            
            # 크롤링 실행 및 청크 저장
            crawler.crawl(max_pages=max_pages, max_workers=max_workers, save_interval=chunk_size)
            crawler.chunked_save(chunk_size=chunk_size)
            
            # 백업 삭제 (이어서 진행이 아닌 경우)
            if not resume and os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            
            print(f"'{name}' 사이트가 성공적으로 업데이트되었습니다.")
            return True
        except Exception as e:
            print(f"업데이트 중 오류 발생: {e}")
            # 실패하면 백업 복원 (이어서 진행이 아닌 경우)
            if not resume:
                if os.path.exists(site_dir):
                    shutil.rmtree(site_dir)
                if os.path.exists(backup_dir):
                    shutil.copytree(backup_dir, site_dir)
                    shutil.rmtree(backup_dir)
            return False
    
    def get_site_dir(self, name):
        """사이트 디렉토리 경로 반환"""
        if name not in self.config["SITES"]:
            return None
        return os.path.join(DOCS_BASE_DIR, name)
    
    def get_markdown_dir(self, name):
        """사이트의 마크다운 디렉토리 경로 반환"""
        site_dir = self.get_site_dir(name)
        if not site_dir:
            return None
        return site_dir


# CLI 인터페이스
def main():
    parser = argparse.ArgumentParser(description="웹사이트 크롤링 및 문서 생성 도구 (OpenAI API 비용 없음)")
    subparsers = parser.add_subparsers(dest="command", help="사용할 명령")
    
    # 사이트 목록
    subparsers.add_parser("list-sites", help="크롤링된 사이트 목록 표시")
    
    # 사이트 추가
    add_parser = subparsers.add_parser("add-site", help="새 사이트 추가 및 크롤링")
    add_parser.add_argument("name", help="사이트 이름 (식별자)")
    add_parser.add_argument("url", help="크롤링 시작 URL")
    add_parser.add_argument("--max-pages", type=int, default=5000, help="크롤링할 최대 페이지 수")
    add_parser.add_argument("--resume", action="store_true", help="이전 크롤링 작업 이어서 진행")
    add_parser.add_argument("--exclusion", action="append", help="제외할 URL 패턴 (정규식, 여러 번 사용 가능)")
    add_parser.add_argument("--delay", type=float, default=0.5, help="요청 간 딜레이 (초)")
    add_parser.add_argument("--chunk-size", type=int, default=1000, help="청크 저장 시 페이지 크기")
    add_parser.add_argument("--workers", type=int, default=5, help="동시 처리 워커 수")
    
    # 사이트 제거
    remove_parser = subparsers.add_parser("remove-site", help="사이트 제거")
    remove_parser.add_argument("name", help="제거할 사이트 이름")
    
    # 사이트 업데이트
    update_parser = subparsers.add_parser("update-site", help="기존 사이트 업데이트")
    update_parser.add_argument("name", help="업데이트할 사이트 이름")
    update_parser.add_argument("--max-pages", type=int, default=5000, help="크롤링할 최대 페이지 수")
    update_parser.add_argument("--resume", action="store_true", help="이전 크롤링 작업 이어서 진행")
    update_parser.add_argument("--exclusion", action="append", help="제외할 URL 패턴 (정규식, 여러 번 사용 가능)")
    update_parser.add_argument("--delay", type=float, default=0.5, help="요청 간 딜레이 (초)")
    update_parser.add_argument("--chunk-size", type=int, default=1000, help="청크 저장 시 페이지 크기")
    update_parser.add_argument("--workers", type=int, default=5, help="동시 처리 워커 수")
    
    # 사이트 열기
    open_parser = subparsers.add_parser("open", help="사이트의 마크다운 문서 디렉토리 열기")
    open_parser.add_argument("name", help="열 사이트 이름")
    
    # 현재 프로젝트에 링크 생성 기능 추가
    link_parser = subparsers.add_parser("link", help="현재 프로젝트 폴더에 사이트 문서 링크 생성")
    link_parser.add_argument("name", help="링크할 사이트 이름")
    link_parser.add_argument("--project-dir", help="프로젝트 디렉토리 (기본값: 현재 디렉토리)")
    link_parser.add_argument("--docs-folder", default="docs", help="프로젝트 내 문서 폴더 이름 (기본값: docs)")
    
    # 모든 사이트를 현재 프로젝트에 링크
    link_all_parser = subparsers.add_parser("link-all", help="모든 사이트를 현재 프로젝트에 링크")
    link_all_parser.add_argument("--project-dir", help="프로젝트 디렉토리 (기본값: 현재 디렉토리)")
    link_all_parser.add_argument("--docs-folder", default="docs", help="프로젝트 내 문서 폴더 이름 (기본값: docs)")
    
    # 대규모 크롤링용 명령 추가
    bulk_parser = subparsers.add_parser("bulk-crawl", help="CSV나 JSON 파일에서 여러 사이트 크롤링")
    bulk_parser.add_argument("file", help="사이트 목록이 있는 CSV 또는 JSON 파일 경로")
    bulk_parser.add_argument("--max-pages", type=int, default=200, help="사이트당 크롤링할 최대 페이지 수")
    bulk_parser.add_argument("--format", choices=["csv", "json"], default="csv", help="입력 파일 형식")
    
    args = parser.parse_args()
    
    # 관리자 인스턴스 생성
    manager = DocsManager()
    
    # 명령 처리
    if args.command == "list-sites":
        manager.list_sites()
    elif args.command == "add-site":
        if args.exclusion:
            exclusion_patterns = args.exclusion
        else:
            # 기본적으로 제외할 패턴 (로그인, 관리자, 검색 등)
            exclusion_patterns = [
                r'/wp-admin/',
                r'/login/',
                r'/search/',
                r'/cart/',
                r'/account/',
                r'\?s=',  # 검색 쿼리
                r'\?p=\d+&',  # 페이지네이션 쿼리
                r'/page/\d+/$'  # 페이지네이션 URL
            ]
        
        manager.add_site(
            args.name,
            args.url,
            max_pages=args.max_pages,
            resume=args.resume,
            exclusion_patterns=exclusion_patterns,
            rate_limit_delay=args.delay,
            chunk_size=args.chunk_size,
            max_workers=args.workers
        )
    elif args.command == "remove-site":
        manager.remove_site(args.name)
    elif args.command == "update-site":
        if args.exclusion:
            exclusion_patterns = args.exclusion
        else:
            # 기본적으로 제외할 패턴
            exclusion_patterns = [
                r'/wp-admin/',
                r'/login/',
                r'/search/',
                r'/cart/',
                r'/account/',
                r'\?s=',
                r'\?p=\d+&',
                r'/page/\d+/$'
            ]
        
        manager.update_site(
            args.name,
            max_pages=args.max_pages,
            resume=args.resume,
            exclusion_patterns=exclusion_patterns,
            rate_limit_delay=args.delay,
            chunk_size=args.chunk_size,
            max_workers=args.workers
        )
    elif args.command == "open":
        markdown_dir = manager.get_markdown_dir(args.name)
        if markdown_dir:
            if sys.platform == "win32":
                os.startfile(markdown_dir)
            elif sys.platform == "darwin":  # macOS
                os.system(f"open '{markdown_dir}'")
            else:  # linux
                os.system(f"xdg-open '{markdown_dir}'")
        else:
            print(f"'{args.name}' 사이트가 존재하지 않습니다.")
    elif args.command == "link":
        # 단일 사이트를 현재 프로젝트에 링크
        create_project_link(manager, args.name, args.project_dir, args.docs_folder)
    elif args.command == "link-all":
        # 모든 사이트를 현재 프로젝트에 링크
        create_all_project_links(manager, args.project_dir, args.docs_folder)
    elif args.command == "bulk-crawl":
        bulk_crawl(manager, args.file, args.max_pages, args.format)
    else:
        parser.print_help()


def create_project_link(manager, site_name, project_dir=None, docs_folder="docs"):
    """현재 프로젝트에 사이트 문서 링크 생성"""
    # 사이트 확인
    if site_name not in manager.config["SITES"]:
        print(f"'{site_name}' 사이트가 존재하지 않습니다.")
        return False
    
    # 프로젝트 디렉토리 결정
    if not project_dir:
        project_dir = os.getcwd()
    
    # 문서 디렉토리 생성
    docs_dir = os.path.join(project_dir, docs_folder)
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    
    # 링크 대상 찾기 - 더 이상 마크다운 디렉토리가 아닌 일반 디렉토리를 사용
    source_dir = manager.get_site_dir(site_name)
    if not source_dir:
        print(f"'{site_name}' 사이트의 디렉토리를 찾을 수 없습니다.")
        return False
    
    # 링크 생성
    link_path = os.path.join(docs_dir, site_name)
    
    # 기존 링크/폴더 제거
    if os.path.exists(link_path):
        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.isdir(link_path):
            shutil.rmtree(link_path)
    
    # 운영체제에 따른 링크 생성
    try:
        if sys.platform == "win32":
            # Windows에서는 junction 또는 symbolic link 사용
            import ctypes
            kdll = ctypes.windll.LoadLibrary("kernel32.dll")
            kdll.CreateSymbolicLinkW(link_path, source_dir, 1)  # 1은 directory flag
        else:
            # Unix 계열에서는 심볼릭 링크
            os.symlink(source_dir, link_path)
        
        print(f"'{site_name}' 사이트가 '{link_path}'에 링크되었습니다.")
        return True
    except Exception as e:
        print(f"링크 생성 중 오류 발생: {e}")
        return False


def create_all_project_links(manager, project_dir=None, docs_folder="docs"):
    """모든 사이트를 현재 프로젝트에 링크"""
    sites = manager.config["SITES"]
    if not sites:
        print("등록된 사이트가 없습니다.")
        return
    
    success_count = 0
    for name in sites:
        if create_project_link(manager, name, project_dir, docs_folder):
            success_count += 1
    
    print(f"총 {len(sites)}개 사이트 중 {success_count}개가 성공적으로 링크되었습니다.")
    
    # index.md 파일 생성
    if not project_dir:
        project_dir = os.getcwd()
    
    docs_dir = os.path.join(project_dir, docs_folder)
    with open(os.path.join(docs_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("# 프로젝트 문서 참조 목록\n\n")
        f.write("이 문서는 프로젝트에서 참조할 수 있는 모든 외부 문서 목록입니다.\n\n")
        
        for name in sites:
            site_dir = os.path.join(docs_dir, name)
            if os.path.exists(site_dir):
                index_file = os.path.join(site_dir, "index.md")
                site_url = sites[name]
                
                f.write(f"## {name}\n\n")
                f.write(f"출처: {site_url}\n\n")
                
                if os.path.exists(index_file):
                    f.write(f"[{name} 문서 목록 보기](./{name}/index.md)\n\n")
                else:
                    f.write(f"{name} 문서 폴더: `./{name}/`\n\n")


# 대규모 크롤링 기능
def bulk_crawl(manager, file_path, max_pages=200, format="csv"):
    """CSV나 JSON 파일에서 여러 사이트 크롤링"""
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return
    
    sites = []
    
    # 파일 형식에 따라 사이트 목록 불러오기
    if format == "csv":
        import csv
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'name' in row and 'url' in row:
                        sites.append({
                            'name': row['name'],
                            'url': row['url'],
                            'max_pages': int(row.get('max_pages', max_pages))
                        })
        except Exception as e:
            print(f"CSV 파일 읽기 오류: {e}")
            return
    else:  # JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sites = json.load(f)
        except Exception as e:
            print(f"JSON 파일 읽기 오류: {e}")
            return
    
    if not sites:
        print("크롤링할 사이트가 없습니다.")
        return
    
    print(f"총 {len(sites)}개 사이트를 크롤링합니다.")
    
    # 진행 현황 파일
    progress_file = os.path.join(os.path.dirname(file_path), "crawl_progress.json")
    
    # 이미 완료된 사이트 확인
    completed = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                completed = set(progress.get('completed', []))
            print(f"이미 {len(completed)}개 사이트 크롤링이 완료되었습니다.")
        except:
            pass
    
    # 각 사이트 크롤링
    success_count = 0
    for i, site in enumerate(sites):
        site_name = site['name']
        
        # 이미 완료된 사이트는 건너뛰기
        if site_name in completed:
            print(f"[{i+1}/{len(sites)}] '{site_name}' - 이미 완료됨, 건너뛰기")
            success_count += 1
            continue
        
        print(f"[{i+1}/{len(sites)}] '{site_name}' 크롤링 중... ({site['url']})")
        
        # 크롤링 시도
        if manager.add_site(site_name, site['url'], site.get('max_pages', max_pages)):
            success_count += 1
            completed.add(site_name)
            
            # 진행 현황 업데이트
            try:
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump({'completed': list(completed)}, f)
            except Exception as e:
                print(f"진행 현황 저장 오류: {e}")
        
        print(f"현재까지 {success_count}/{i+1} 사이트 완료 (전체 {len(sites)}개 중)")
    
    print(f"\n크롤링 완료: 총 {len(sites)}개 사이트 중 {success_count}개 성공")


if __name__ == "__main__":
    main()