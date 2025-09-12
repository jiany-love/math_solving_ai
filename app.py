"""Flask 웹 애플리케이션 엔트리포인트.

업로드된 수학 문제(이미지)를 수신 → 전처리/영역 검출/간단 솔버 실행 →
생성된 결과 이미지와 로그를 정리하여 템플릿에 전달한다.

주요 흐름:
1) 사용자가 이미지를 업로드 (POST /)
2) 파일 확장자 검증 및 저장 (타임스탬프 붙여 충돌 방지)
3) 간단 OCR & 수학 처리(SimpleMathSolver) 시도 (미설치 환경 허용)
4) 고급 영역 검출(get_math_regions) 수행 (전처리/라인 OCR)
5) 생성된 중간 산출물(전처리, 영역 시각화)을 결과 디렉터리로 복사
6) Solver 로그 + 전체 OCR 텍스트를 로그 파일로 저장
7) 결과를 index.html 렌더링 컨텍스트로 반환

에러/예외 상황도 사용자에게 flash 메시지 또는 안내 텍스트로 전달한다.
"""

import os
import io
import sys
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# 백엔드 폴더는 수정하지 않고 import 사용
from backend.image import get_math_regions

app = Flask(__name__)
app.secret_key = "math-ai-secret"  # 세션/flash 사용을 위한 시크릿 키 (실서비스에서는 환경변수 관리 권장)

###############################################
# 디렉터리/환경 설정
###############################################
UPLOAD_DIR = os.path.join('static', 'uploads')   # 업로드 원본 저장
RESULT_DIR = os.path.join('static', 'results')   # 파생 결과(전처리/검출) 저장
LOG_DIR = os.path.join('static', 'logs')         # 실행 로그 저장
for d in (UPLOAD_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)  # 재실행 시에도 안전

ALLOWED_EXT = {"png", "jpg", "jpeg"}  # 허용 확장자 (추가 필요 시 확장)

def allowed_file(filename: str) -> bool:
    """업로드 파일 확장자 검증.

    점(.) 존재 여부와 허용 목록(소문자) 매칭으로 간단히 판단.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def capture_stdout(func, *args, **kwargs):
    """주어진 함수 실행 중 print 출력(stdout)을 문자열로 캡처.

    Solver 실행 결과(로그)를 별도로 저장하거나 템플릿에 표시하기 위해 사용.
    예외 발생 시 메시지를 캡처 버퍼에 기록하고 정상적으로 stdout 복구.
    """
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"[오류] {e}")
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()

def run_simple_solver(image_path: str) -> str:
    """간단 솔버(SimpleMathSolver) 실행 후 stdout 로그 반환.

    lazy import: torch/easyocr 미설치 환경에서도 전체 앱이 죽지 않도록 import 시도 후 실패를 안내.
    """
    try:
        from backend.simple_math_solver import SimpleMathSolver  # type: ignore
    except Exception as e:
        return (
            f"[안내] SimpleMathSolver를 사용할 수 없습니다: {e}\n"
            "(torch/easyocr 미설치 환경에서는 이 메시지가 정상입니다)"
        )
    solver = SimpleMathSolver()
    return capture_stdout(solver.extract_and_solve, image_path)

def run_ocr_regions(image_path: str):
    """고급 영역 검출 + OCR 결과 획득.

    `get_math_regions` 내부에서 전처리 및 영역 시각화 이미지를 생성.
    반환: (수학/텍스트 영역 리스트, 전체 누적 텍스트)
    """
    math_regions, full_text = get_math_regions(image_path)
    return math_regions, full_text

def collect_result_images(source_dir: str, timestamp: str) -> list:
    """생성된 파생 이미지들을 결과 폴더로 수집.

    Simple / 고급 전처리 및 검출 과정에서 원본 업로드 경로에 만들어진 파일들을
    타임스탬프 prefix를 붙여 `RESULT_DIR`로 복사 (UI에서 쉽게 참조하기 위함).
    누락 가능성(환경에 따라 일부 파일만 생성)에 대비하여 존재 여부 확인 후 선택 복사.
    """
    candidates = [
        'processed_problem.png',     # 고급 전처리 결과
        'detected_regions.png',      # 영역 검출 시각화
        'processed_simple.png',      # 간단 솔버 전처리 결과
    ]
    collected = []
    for name in candidates:
        src_path = os.path.join(source_dir, name)
        if os.path.exists(src_path):  # 파일이 실제 생성된 경우만 복사
            dst_name = f"{timestamp}_{name}"
            dst_path = os.path.join(RESULT_DIR, dst_name)
            shutil.copy2(src_path, dst_path)
            collected.append(dst_name)
    return collected

@app.route('/', methods=['GET', 'POST'])
def index():
    """메인 페이지: 이미지 업로드 및 처리 결과 렌더링.

    POST 처리 시 주 흐름:
      - 파일 존재/이름/확장자 검증
      - 저장 후 두 종류 처리 실행(간단/고급)
      - 파생 이미지 수집 & 로그 파일 생성
      - 템플릿에 필요한 모든 경로/텍스트 전달
    """
    context = {}
    if request.method == 'POST':
        # 1) 폼 필드 존재 검사
        if 'file' not in request.files:
            flash('파일이 포함되지 않았습니다.')
            return redirect(request.url)
        file = request.files['file']

        # 2) 파일명 비어있는지 확인 (사용자가 선택 안 했을 때)
        if file.filename == '':
            flash('선택된 파일이 없습니다.')
            return redirect(request.url)

        # 3) 확장자 검증 후 처리
        if file and allowed_file(file.filename):
            basename = secure_filename(file.filename)  # 보안상 unsafe 문자 제거
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{basename}"  # 중복 방지
            upload_path = os.path.join(UPLOAD_DIR, filename)
            file.save(upload_path)

            # 4) Solver 실행 (각각 독립적으로 실패 허용 가능)
            simple_log = run_simple_solver(upload_path)
            math_regions, full_text = run_ocr_regions(upload_path)

            # 5) 결과 이미지 수집 (업로드 경로 기준)
            result_images = collect_result_images(os.path.dirname(upload_path), timestamp)

            # 6) 로그 파일 생성 (solver 로그 + 전체 텍스트)
            log_filename = f"log_{timestamp}.txt"
            log_path = os.path.join(LOG_DIR, log_filename)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(simple_log)
                f.write('\n\n[전체 OCR 텍스트]\n')
                f.write(full_text or '(없음)')

            # 7) 템플릿에 전달할 컨텍스트 구성
            context.update({
                'uploaded_image': os.path.join('uploads', filename),
                'result_images': [os.path.join('results', r) for r in result_images],
                'solver_log': simple_log,
                'ocr_text': full_text,
                'math_regions': math_regions,
                'log_file': os.path.join('logs', log_filename)
            })
        else:
            flash('허용되지 않는 파일 형식입니다.')
    return render_template('index.html', **context)

@app.route('/logs/<path:logname>')
def get_log(logname):
    """저장된 로그 파일 다운로드/열람 라우트."""
    return send_from_directory(LOG_DIR, logname, as_attachment=True)

if __name__ == '__main__':
    # 개발 편의를 위해 debug=True (운영 환경에서는 False + WSGI 서버 사용 권장)
    app.run(debug=True, host='0.0.0.0', port=5000)
