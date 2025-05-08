# 웹사이트 크롤링 및 문서 참조 도구 사용 가이드

이 도구는 웹사이트를 크롤링하여 마크다운 문서로 변환하고, GitHub Copilot이 참조할 수 있도록 해주는 유틸리티입니다.

## 설치 및 준비

1. **파이썬 크롤링 파일 위치**
   - `website_crawler.py` 파일을 iCloud Drive에 저장하세요.
   - 추천 위치: `~/Library/Mobile Documents/com~apple~CloudDocs/Tools/website_crawler.py`
   - 이렇게 하면 모든 Apple 기기에서 동일한 스크립트를 사용할 수 있습니다.

2. **가상환경 설정 및 패키지 설치 (권장)**
   
   스크립트가 어디에 저장되어 있든, 패키지는 로컬 시스템이나 가상환경에 설치됩니다. 가장 좋은 방법은 가상환경을 만들어 사용하는 것입니다:

   ```bash
   # 가상환경을 위한 디렉토리 생성 (한 번만 실행)
   mkdir -p ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/venv

   # 가상환경 생성 (한 번만 실행)
   python -m venv ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/venv/crawler_env

   # 가상환경 활성화 (터미널 세션마다 실행)
   source ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/venv/crawler_env/bin/activate

   # 필요한 패키지 설치 (가상환경 생성 후 한 번만 실행)
   pip install requests beautifulsoup4 tqdm
   ```

   각 컴퓨터에서 처음 사용할 때는 가상환경을 활성화하고 필요한 패키지를 설치해야 합니다. 이후에는 가상환경을 활성화만 하면 됩니다.

3. **시스템 전체 설치 (대안)**
   
   모든 프로젝트에서 사용하려면 시스템 전체에 패키지를 설치할 수도 있습니다:
   
   ```bash
   pip install requests beautifulsoup4 tqdm
   ```

## 기본 사용법

### 0. 가상환경 활성화 (가상환경을 사용하는 경우)

```bash
source ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/venv/crawler_env/bin/activate
```

### 1. 웹사이트 크롤링하기

```bash
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py add-site 사이트이름 https://example.com/docs/ --max-pages 100
```

- `사이트이름`: 크롤링한 사이트를 구분할 이름 (영문/숫자 사용 권장)
- `https://example.com/docs/`: 크롤링을 시작할 URL
- `--max-pages`: 최대 크롤링할 페이지 수 (기본값: 50)

예시:
```bash
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py add-site kadence https://www.kadencewp.com/help-center/
```

### 2. 크롤링된 사이트 목록 확인

```bash
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py list-sites
```

### 3. 프로젝트에 문서 링크 생성하기

프로젝트 디렉토리에서 다음 명령을 실행합니다:

```bash
cd ~/GitHub/work-space/homepage\ v2
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py link-all
```

이 명령은 프로젝트 내에 `docs` 폴더를 생성하고, 그 안에 모든 크롤링된 사이트를 링크합니다.

## 각 프로젝트에서 사용하는 절차

새 프로젝트를 시작하거나 기존 프로젝트에서 문서를 참조하려면:

1. **프로젝트 디렉토리로 이동**
   ```bash
   cd 프로젝트경로
   ```

2. **모든 사이트 문서를 프로젝트에 링크**
   ```bash
   python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py link-all
   ```

3. **이제 GitHub Copilot이 문서를 참조할 수 있습니다**
   - 문서 관련 질문을 하면 Copilot이 링크된 문서를 참조하여 답변합니다

## 고급 사용법

### 특정 사이트만 링크하기

```bash
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py link 사이트이름
```

### 사이트 문서 업데이트하기

```bash
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py update-site 사이트이름 --max-pages 100
```

### 문서 폴더 직접 열기

```bash
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py open 사이트이름
```

### 사이트 제거하기

```bash
python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py remove-site 사이트이름
```

## 저장 위치

- 크롤링 도구: iCloud Drive의 `Tools` 폴더에 저장됩니다
- 크롤링 문서: iCloud Drive의 `DocsSearch` 폴더에 저장됩니다
- iCloud Drive가 없는 경우 홈 디렉토리의 `.docsearch` 폴더에 저장됩니다
- 저장 위치는 모든 Apple 기기 간에 동기화되어 어디서나 같은 도구와 문서를 사용할 수 있습니다

## 단축 명령어 설정 (선택사항)

터미널에서 더 쉽게 사용하기 위해 배시 별칭(alias)을 설정할 수 있습니다:

1. `~/.zshrc` 또는 `~/.bash_profile` 파일을 열고 다음 줄을 추가합니다:

```bash
alias docscrawl='python ~/Library/Mobile\ Documents/com~apple~CloudDocs/Tools/website_crawler.py'
```

2. 파일을 저장하고 다음 명령으로 변경사항을 적용합니다:

```bash
source ~/.zshrc  # 또는 source ~/.bash_profile
```

3. 이제 다음과 같이 더 간단하게 명령을 실행할 수 있습니다:

```bash
docscrawl add-site example https://example.com
docscrawl link-all
```