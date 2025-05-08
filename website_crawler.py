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
import markdown
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

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
    def __init__(self, start_url, output_dir="crawled_content", site_name=None):
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.visited_urls = set()
        self.content_data = {}
        self.output_dir = output_dir
        self.site_name = site_name or self.base_domain
        
        # 출력 디렉토리 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 마크다운 디렉토리 생성
        self.markdown_dir = os.path.join(output_dir, "markdown")
        if not os.path.exists(self.markdown_dir):
            os.makedirs(self.markdown_dir)
    
    def is_valid_url(self, url):
        """URL이 유효한지, 같은 도메인에 속하는지 확인"""
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == self.base_domain
    
    def get_page_content(self, url):
        """페이지 내용 가져오기"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            if response.status_code == 200:
                return response.text
            return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_links(self, url, html_content):
        """페이지에서 링크 추출"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(url, href)
            
            # 같은 도메인이고 아직 방문하지 않은 URL만 추가
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)
        
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
    
    def crawl(self, max_pages=50, max_workers=5):
        """웹사이트 크롤링 시작"""
        queue = [self.start_url]
        page_count = 0
        
        while queue and page_count < max_pages:
            # 여러 URL을 병렬로 처리
            batch_size = min(max_workers, len(queue), max_pages - page_count)
            current_batch = queue[:batch_size]
            queue = queue[batch_size:]
            
            # 결과 저장할 임시 딕셔너리
            new_results = {}
            new_links = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def process_url(url):
                    if url in self.visited_urls:
                        return None, []
                    
                    print(f"Crawling: {url}")
                    self.visited_urls.add(url)
                    
                    html_content = self.get_page_content(url)
                    if not html_content:
                        return None, []
                    
                    # 내용 추출
                    page_data = self.extract_content(url, html_content)
                    
                    # 링크 추출
                    links = self.extract_links(url, html_content)
                    
                    # 서버 부하 방지를 위한 딜레이
                    time.sleep(0.5)
                    
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
                        print(f"Error processing {url}: {e}")
            
            # 새 결과 및 링크 업데이트
            self.content_data.update(new_results)
            queue.extend([link for link in new_links if link not in self.visited_urls])
        
        self.save_content()
        return self.content_data
    
    def save_content(self):
        """크롤링한 내용 저장"""
        # JSON으로 저장
        with open(os.path.join(self.output_dir, 'crawled_data.json'), 'w', encoding='utf-8') as f:
            json.dump(self.content_data, f, ensure_ascii=False, indent=2)
        
        # 마크다운 파일로 저장
        for page_id, page_data in self.content_data.items():
            # 파일명에 사용할 수 없는 문자 제거
            safe_title = re.sub(r'[^\w\s-]', '', page_data['title']).strip()
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"{page_id}_{safe_title[:50]}.md"
            
            with open(os.path.join(self.markdown_dir, filename), 'w', encoding='utf-8') as f:
                # YAML 프론트매터 추가
                f.write(f"---\n")
                f.write(f"title: \"{page_data['title']}\"\n")
                f.write(f"url: {page_data['url']}\n")
                f.write(f"date: {time.strftime('%Y-%m-%d')}\n")
                f.write(f"source: \"{self.site_name}\"\n")
                f.write(f"---\n\n")
                
                # 마크다운 내용 작성
                f.write(f"# {page_data['title']}\n\n")
                
                current_level = 1
                for item in page_data['content']:
                    if item['tag'].startswith('h'):
                        level = int(item['tag'][1:])
                        # 제목 수준 조정 (h1 제목이 이미 있으므로)
                        adjusted_level = max(1, level)
                        f.write(f"{'#' * adjusted_level} {item['text']}\n\n")
                        current_level = adjusted_level
                    elif item['tag'] == 'p':
                        f.write(f"{item['text']}\n\n")
                    elif item['tag'] == 'li':
                        f.write(f"- {item['text']}\n")
                    elif item['tag'] in ['code', 'pre']:
                        f.write(f"```\n{item['text']}\n```\n\n")
                
                # 출처 정보 추가
                f.write(f"\n\n---\n")
                f.write(f"Source: [{page_data['url']}]({page_data['url']})\n")
        
        # 인덱스 파일 생성 (모든 페이지 링크 포함)
        with open(os.path.join(self.markdown_dir, 'index.md'), 'w', encoding='utf-8') as f:
            f.write(f"# {self.site_name} Documentation\n\n")
            f.write(f"This documentation was automatically crawled from {self.start_url} on {time.strftime('%Y-%m-%d')}.\n\n")
            f.write(f"## Pages\n\n")
            
            for page_id, page_data in self.content_data.items():
                safe_title = re.sub(r'[^\w\s-]', '', page_data['title']).strip()
                safe_title = re.sub(r'[-\s]+', '-', safe_title)
                filename = f"{page_id}_{safe_title[:50]}.md"
                f.write(f"- [{page_data['title']}](./{filename})\n")
                
        print(f"크롤링 완료: {len(self.content_data)} 페이지가 {self.markdown_dir}에 저장되었습니다.")


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
    
    def add_site(self, name, url, max_pages=50):
        """새 사이트 추가 및 크롤링"""
        # 이름 검증
        if name in self.config["SITES"]:
            print(f"'{name}' 사이트가 이미 존재합니다. 다른 이름을 사용하세요.")
            return False
        
        # 사이트 디렉토리 경로
        site_dir = os.path.join(DOCS_BASE_DIR, name)
        
        # 크롤링 실행
        try:
            crawler = WebsiteCrawler(url, output_dir=site_dir, site_name=name)
            crawler.crawl(max_pages=max_pages)
            
            # 설정 파일에 사이트 추가
            self.config["SITES"][name] = url
            self._save_config()
            
            print(f"'{name}' 사이트가 성공적으로 크롤링되었습니다.")
            print(f"문서는 {os.path.join(site_dir, 'markdown')} 디렉토리에 저장되었습니다.")
            return True
        except Exception as e:
            print(f"크롤링 중 오류 발생: {e}")
            # 실패한 경우 디렉토리 정리
            if os.path.exists(site_dir):
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
    
    def update_site(self, name, max_pages=50):
        """기존 사이트 업데이트"""
        if name not in self.config["SITES"]:
            print(f"'{name}' 사이트가 존재하지 않습니다.")
            return False
        
        url = self.config["SITES"][name]
        site_dir = os.path.join(DOCS_BASE_DIR, name)
        
        # 기존 데이터 백업
        backup_dir = f"{site_dir}_backup_{int(time.time())}"
        if os.path.exists(site_dir):
            shutil.copytree(site_dir, backup_dir)
        
        # 재크롤링
        try:
            # 기존 디렉토리 삭제
            if os.path.exists(site_dir):
                shutil.rmtree(site_dir)
            
            # 새로 크롤링
            crawler = WebsiteCrawler(url, output_dir=site_dir, site_name=name)
            crawler.crawl(max_pages=max_pages)
            
            # 백업 삭제
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            
            print(f"'{name}' 사이트가 성공적으로 업데이트되었습니다.")
            return True
        except Exception as e:
            print(f"업데이트 중 오류 발생: {e}")
            # 실패하면 백업 복원
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
        return os.path.join(site_dir, "markdown")


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
    add_parser.add_argument("--max-pages", type=int, default=50, help="크롤링할 최대 페이지 수")
    
    # 사이트 제거
    remove_parser = subparsers.add_parser("remove-site", help="사이트 제거")
    remove_parser.add_argument("name", help="제거할 사이트 이름")
    
    # 사이트 업데이트
    update_parser = subparsers.add_parser("update-site", help="기존 사이트 업데이트")
    update_parser.add_argument("name", help="업데이트할 사이트 이름")
    update_parser.add_argument("--max-pages", type=int, default=50, help="크롤링할 최대 페이지 수")
    
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
    
    args = parser.parse_args()
    
    # 관리자 인스턴스 생성
    manager = DocsManager()
    
    # 명령 처리
    if args.command == "list-sites":
        manager.list_sites()
    elif args.command == "add-site":
        manager.add_site(args.name, args.url, args.max_pages)
    elif args.command == "remove-site":
        manager.remove_site(args.name)
    elif args.command == "update-site":
        manager.update_site(args.name, args.max_pages)
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
    
    # 링크 대상 찾기
    source_dir = manager.get_markdown_dir(site_name)
    if not source_dir:
        print(f"'{site_name}' 사이트의 마크다운 디렉토리를 찾을 수 없습니다.")
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


if __name__ == "__main__":
    main()