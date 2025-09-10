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
app.secret_key = "math-ai-secret"

# 디렉터리 설정
UPLOAD_DIR = os.path.join('static', 'uploads')
RESULT_DIR = os.path.join('static', 'results')
LOG_DIR = os.path.join('static', 'logs')
for d in (UPLOAD_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}

def allowed_file(filename: str) -> bool: 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def capture_stdout(func, *args, **kwargs):
    """함수 실행 중 stdout 캡처"""
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
    try:
        # lazy import to avoid ImportError when easyocr/torch aren't installed
        from backend.simple_math_solver import SimpleMathSolver  # type: ignore
    except Exception as e:
        return f"[안내] SimpleMathSolver를 사용할 수 없습니다: {e}\n(torch/easyocr 미설치 환경에서는 이 메시지가 정상입니다)"
    solver = SimpleMathSolver()
    return capture_stdout(solver.extract_and_solve, image_path)

def run_ocr_regions(image_path: str):
    # get_math_regions 내부에서 전처리/검출 이미지를 생성합니다.
    math_regions, full_text = get_math_regions(image_path)
    return math_regions, full_text

def collect_result_images(source_dir: str, timestamp: str) -> list:
    """업로드 디렉터리(이미지와 동일 경로)에 생성된 결과 이미지를 결과 폴더로 복사합니다."""
    candidates = [
        'processed_problem.png',
        'detected_regions.png',
        'processed_simple.png',
    ]
    collected = []
    for name in candidates:
        src_path = os.path.join(source_dir, name)
        if os.path.exists(src_path):
            dst_name = f"{timestamp}_{name}"
            dst_path = os.path.join(RESULT_DIR, dst_name)
            shutil.copy2(src_path, dst_path)
            collected.append(dst_name)
    return collected

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('파일이 포함되지 않았습니다.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('선택된 파일이 없습니다.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            basename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{basename}"
            upload_path = os.path.join(UPLOAD_DIR, filename)
            file.save(upload_path)

            # Solver 실행
            simple_log = run_simple_solver(upload_path)
            math_regions, full_text = run_ocr_regions(upload_path)

            # 결과 이미지 수집 (업로드 폴더 기준)
            result_images = collect_result_images(os.path.dirname(upload_path), timestamp)

            # 로그 저장
            log_filename = f"log_{timestamp}.txt"
            log_path = os.path.join(LOG_DIR, log_filename)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(simple_log)
                f.write('\n\n[전체 OCR 텍스트]\n')
                f.write(full_text or '(없음)')
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
    return send_from_directory(LOG_DIR, logname, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
