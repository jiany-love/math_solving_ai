"""image.py - 개선된 OCR 이미지 처리 모듈

핵심 기능:
1. 전처리 (노이즈 제거, 대비 향상, 이진화, 모폴로지)로 텍스트 대비 극대화
2. MSER + 윤곽선 기반의 하이브리드 텍스트 영역 검출 후 IoU 기반 중복 제거
3. 각 검출 영역에 대해 Tesseract OCR 적용 (한/영 혼용 허용)
4. 검출 결과를 영역 정보(list[dict])와 전체 텍스트로 구조화 반환
5. 시각화 이미지(전처리 결과, 검출 영역 박스)를 파일로 저장하여 프론트엔드가 활용 가능

주요 설계 메모:
- pytesseract 미설치 환경에서도 모듈 import 자체는 실패하지 않도록 try/except 처리
- 경량화 목적으로 EasyOCR 등 무거운 의존성 없이 Tesseract 위주 사용
- 작은 영역 노이즈를 제거하면서도 수식 내 기호 깨짐을 최소화하기 위해 커널 크기를 작게 유지

반환 데이터 예시 (get_math_regions):
[
    { 'line_number': 1, 'region': (x,y,w,h), 'area': int, 'text': '2x + 3' },
    ...
], "2x + 3 ..." (전체 텍스트)
"""

import cv2
import numpy as np
import os
try:
    import pytesseract
    # Windows의 경우 tesseract 경로 설정 (필요시)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract가 설치되지 않았습니다. OCR 기능을 사용하려면 설치해주세요:")
    print("   pip install pytesseract")
    print("   그리고 Tesseract OCR 엔진도 설치해야 합니다.")

class ImageProcessor:
    """OCR 전처리 + 영역 검출 + 텍스트 추출 담당 클래스."""

    def __init__(self):
        print("[설정] 이미지 처리기 초기화")
        if not TESSERACT_AVAILABLE:
            print("⚠️ OCR 기능이 비활성화되었습니다.")
    
    def preprocess_image(self, image):
        """기본 전처리 파이프라인 (검출용).

        Steps:
          1) Grayscale
          2) Gaussian Blur (경미한 노이즈 제거)
          3) Adaptive Threshold (불균일 조명 보정)
          4) Morph Close (문자 영역 뭉침 강화)
          5) Morph Open (작은 노이즈 제거)
        """
        print("[처리] 이미지 전처리 시작")
        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # 2. 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # 3. 적응적 이진화 (더 나은 텍스트 분리)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        # 4. 텍스트 영역 강화를 위한 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # 5. 작은 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel)
        return cleaned
    
    def preprocess_for_ocr(self, image):
        """OCR 정확도 향상을 위한 고급 전처리.

        - 크기 확대: 작은 글자 세부 정보 확보
        - CLAHE: 지역 대비 향상
        - Otsu Threshold: 전역 이진화로 뚜렷한 경계 확보
        """
        # 1. 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 2. 이미지 크기 증가 (OCR 정확도 향상)
        height, width = gray.shape
        scale_factor = 2 if min(height, width) < 500 else 1.5
        scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                          interpolation=cv2.INTER_CUBIC)
        
        # 3. 대비 개선
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(scaled)
        
        # 4. 가우시안 블러 (약간만)
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # 5. 이진화 (OCR에 최적화)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text_from_region(self, image, bbox):
        """개별 영역 OCR.

        bbox: (x, y, w, h)
        패딩을 더해 경계 클리핑으로 인한 글자 손실을 줄인다.
        """
        if not TESSERACT_AVAILABLE:
            return "[OCR 불가] pytesseract가 설치되지 않음"
        
        x, y, w, h = bbox
        
        # 영역 추출 (약간의 패딩 추가)
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        roi = image[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            return ""
        
        try:
            # OCR을 위한 전처리
            processed_roi = self.preprocess_for_ocr(roi)
            
            # OCR 실행 (한글과 영어 지원)
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-×÷=()[]{}ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz가-힣 '
            text = pytesseract.image_to_string(processed_roi, lang='kor+eng', config=config)
            
            # 텍스트 정리
            cleaned_text = text.strip().replace('\n', ' ').replace('\r', '')
            cleaned_text = ' '.join(cleaned_text.split())  # 여러 공백을 하나로
            
            return cleaned_text
            
        except Exception as e:
            print(f"[오류] OCR 처리 중 오류: {str(e)}")
            return f"[OCR 오류] {str(e)}"
    
    def find_text_regions_advanced(self, preprocessed, original_image):
        """하이브리드 텍스트 영역 검출 (MSER + Contours).

        - 두 기법의 후보를 통합 후 IoU 기반 중복 제거
        - 지나치게 크거나 작은 영역 필터링
        - y, x 순 정렬로 문서 위→아래 흐름 유지
        """
        regions = []
        
        # 1. MSER (Maximally Stable Extremal Regions) 검출기 사용
        mser = cv2.MSER_create()
        regions_mser, _ = mser.detectRegions(preprocessed)
        
        # 2. 윤곽선 기반 검출
        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 3. 두 방법의 결과를 통합
        all_regions = []
        
        # MSER 영역 처리
        for region in regions_mser:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            x, y, w, h = cv2.boundingRect(hull)
            
            # 크기 필터링
            if 15 <= w <= original_image.shape[1] * 0.8 and 10 <= h <= original_image.shape[0] * 0.3:
                all_regions.append((x, y, w, h, w * h))
        
        # 윤곽선 영역 처리
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 크기 필터링 (더 관대한 조건)
            if 15 <= w <= original_image.shape[1] * 0.9 and 10 <= h <= original_image.shape[0] * 0.4:
                all_regions.append((x, y, w, h, w * h))
        
        # 4. 중복 제거 및 정렬
        unique_regions = []
        for region in all_regions:
            x, y, w, h, area = region
            
            # 기존 영역과의 중복 체크
            is_duplicate = False
            for existing in unique_regions:
                ex_x, ex_y, ex_w, ex_h = existing['bbox']
                
                # IoU (Intersection over Union) 계산
                x1 = max(x, ex_x)
                y1 = max(y, ex_y)
                x2 = min(x + w, ex_x + ex_w)
                y2 = min(y + h, ex_y + ex_h)
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = area + (ex_w * ex_h) - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.5:  # 50% 이상 겹치면 중복으로 간주
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area
                })
        
        # 5. Y 좌표 기준으로 정렬 (위에서 아래로)
        unique_regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        return unique_regions
    
    def extract_text(self, image_path):
        """전체 처리 메인 함수.

        반환: OCR 결과 리스트 (각 항목: bbox, text, area, line_number)
        실패 시 빈 리스트 반환 (상위 호출부 안정성 확보)
        """
        try:
            # 이미지 로드 (한글 경로 지원)
            img_array = np.fromfile(image_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise Exception("이미지를 불러올 수 없습니다.")
            
            print(f"[정보] 이미지 크기: {image.shape[1]}x{image.shape[0]}")
            
            # 전처리
            preprocessed = self.preprocess_image(image)
            
            # 결과 저장
            current_dir = os.path.dirname(image_path)
            processed_path = os.path.join(current_dir, "processed_problem.png")
            cv2.imwrite(processed_path, preprocessed)
            print(f"[결과] 전처리된 이미지: {processed_path}")
            
            # 텍스트 영역 검출
            regions = self.find_text_regions_advanced(preprocessed, image)
            
            print(f"[결과] {len(regions)}개의 텍스트 영역을 검출했습니다.")
            
            # 각 영역에서 OCR 실행
            ocr_results = []
            for i, region in enumerate(regions):
                bbox = region['bbox']
                text = self.extract_text_from_region(image, bbox)
                
                if text.strip():  # 빈 텍스트가 아닌 경우만
                    ocr_results.append({
                        'bbox': bbox,
                        'text': text,
                        'area': region['area'],
                        'line_number': i + 1
                    })
                    print(f"[OCR {i+1}] {text}")
            
            # 결과 시각화
            result_image = image.copy()
            for i, result in enumerate(ocr_results):
                x, y, w, h = result['bbox']
                
                # 영역 표시
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 번호 표시
                cv2.putText(result_image, str(result['line_number']), (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 시각화 결과 저장
            detected_path = os.path.join(current_dir, "detected_regions.png")
            cv2.imwrite(detected_path, result_image)
            print(f"[결과] 검출된 영역 이미지: {detected_path}")
            
            return ocr_results
            
        except Exception as e:
            print(f"[오류] 이미지 처리 중 오류 발생: {str(e)}")
            return []
    
    def extract_math_expressions(self, ocr_results):
        """OCR 결과에서 수식 표현식 정보 추출"""
        expressions = []
        all_text = ""
        
        for result in ocr_results:
            expressions.append({
                'line_number': result['line_number'],
                'region': result['bbox'],
                'area': result['area'],
                'text': result['text']
            })
            all_text += result['text'] + " "
        
        print(f"[정보] {len(expressions)}개의 수식 영역을 준비했습니다.")
        print(f"[전체 텍스트] {all_text.strip()}")
        
        return expressions, all_text.strip()


# 함수형 인터페이스 (간편 사용용)
def process_image(image_path):
    """파일 경로 단위의 간단 함수형 헬퍼.

    내부적으로 `ImageProcessor` 인스턴스를 생성해 `extract_text` 호출.
    테스트/재사용 편의성을 위해 제공.
    """
    processor = ImageProcessor()
    return processor.extract_text(image_path)


def get_math_regions(image_path):
    """수학(텍스트) 영역 정보 및 전체 텍스트 반환.

    빈 결과일 경우 ([], "") 형식 보장 → 호출 측에서 null 체크 로직 단순화.
    """
    processor = ImageProcessor()
    ocr_results = processor.extract_text(image_path)
    if ocr_results:
        return processor.extract_math_expressions(ocr_results)
    return [], ""


# 테스트용 메인 함수
if __name__ == "__main__":
    print("개선된 OCR 이미지 처리 모듈 테스트")
    print("="*50)
    
    if not TESSERACT_AVAILABLE:
        print("❌ OCR 기능을 사용하려면 다음을 설치해야 합니다:")
        print("1. pip install pytesseract")
        print("2. Tesseract OCR 엔진 설치")
        print("   - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-kor")
        print("   - macOS: brew install tesseract tesseract-lang")
        print()
    
    # 테스트할 이미지 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(current_dir, "example.png")
    
    if not os.path.exists(test_image):
        print(f"❌ 테스트 이미지가 없습니다: {test_image}")
        print("   example.png 파일을 같은 폴더에 넣어주세요.")
    else:
        print(f"📂 테스트 이미지: {os.path.basename(test_image)}")
        
        # 함수형 인터페이스 테스트
        print("\n🔍 OCR 텍스트 추출 테스트:")
        ocr_results = process_image(test_image)
        
        if ocr_results:
            print(f"✅ {len(ocr_results)}개 영역에서 OCR 성공!")
            for i, result in enumerate(ocr_results[:5]):  # 최대 5개만 출력
                print(f"   영역 {result['line_number']}: '{result['text']}'")
        else:
            print("❌ OCR 결과 없음")
        
        print("\n🧮 수학 영역 분석 테스트:")
        math_regions, full_text = get_math_regions(test_image)
        
        if math_regions:
            print(f"✅ {len(math_regions)}개 수학 영역 분석 완료!")
            print(f"📝 전체 텍스트: '{full_text}'")
            
            for region in math_regions[:3]:  # 최대 3개만 출력
                print(f"   줄 {region['line_number']}: '{region['text']}'")
        else:
            print("❌ 수학 영역 분석 실패")
        
        print(f"\n📁 생성된 파일들:")
        print(f"   • processed_problem.png (전처리된 이미지)")
        print(f"   • detected_regions.png (검출된 영역 표시)")