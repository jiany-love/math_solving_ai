# Math Solving AI - Web Frontend

이 프로젝트는 이미지 기반 수학 문제를 업로드하면 OCR + 간단한 수학/기하 분석을 수행하는 Python 기반 웹 서비스입니다.

## 구성
- backend/: 기존 수학/이미지 처리 로직 (수정 금지 권장)
- app.py: Flask 웹 서버
- static/uploads: 업로드된 원본 이미지 저장
- static/results: 처리된 결과 이미지 (검출/전처리 등)
- static/logs: Solver 실행 로그 저장
- templates/index.html: 메인 UI

## 설치 & 실행
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

브라우저에서: http://127.0.0.1:5000

## 기능
- PNG/JPG 업로드
- EasyOCR 기반 텍스트 추출
- 텍스트 영역 감지 (processed_problem.png / detected_regions.png 생성)
- SimpleMathSolver 로그 표시 및 다운로드
- OCR 전체 텍스트 및 추출 영역 테이블 표시

## 주의
- pytesseract 사용 시 시스템에 Tesseract OCR 엔진 설치 필요
	- macOS: `brew install tesseract tesseract-lang`
- backend 폴더는 로직 의존성이 있으므로 경로 이동/이름 변경 금지
- EasyOCR(=torch 의존) 사용은 선택 사항입니다. 설치를 원하면:
	- 파이썬 3.8–3.12 환경 권장 (3.13은 아직 torch 미지원)
	- `pip install easyocr` 실행 (자동으로 torch 설치 시도)
	- 만약 torch 설치 에러가 나면, 파이썬 버전을 3.12로 낮추거나, 플랫폼에 맞는 torch를 수동 설치하세요: https://pytorch.org/get-started/locally/
	- EasyOCR 없이도 Tesseract 기반 OCR 경로로 동작합니다.

## 향후 개선 아이디어
- 비동기 처리 (Celery + Redis)
- 수식 LaTeX 렌더링
- 고급 MathSolver 통합 (현재 simple solver 사용)
- 다국어 UI 지원

즐거운 해킹 되세요 🚀
