"""simple_math_solver.py

경량 수학 문제 처리기:
    - EasyOCR 로 이미지에서 한/영 텍스트 추출
    - 정규식 기반으로 등식/연산/좌표/숫자 패턴 탐지
    - 간단 계산 및 거리 계산 수행 후 콘솔 출력

특징/제한:
    - 안전하지 않은 eval 사용 (프로토타입 용도)
    - 수식 파싱/오차 처리 단순
    - torch/easyocr 미설치 환경에서는 상위(app.py)에서 import 예외 처리
"""

import os
import cv2
import numpy as np
import easyocr
import re

class SimpleMathSolver:
    """가벼운 OCR + 규칙 기반 수학 분석기."""

    def __init__(self):
        print("간단한 수학 문제 풀이 시스템")
        print("="*50)
        # EasyOCR 초기화 (모델 다운로드 캐싱 후 재사용)
        print("[설정] OCR 초기화 중...")
        self.reader = easyocr.Reader(['ko', 'en'])
        print("[설정] OCR 초기화 완료")

    def preprocess_image(self, image):
        """OCR 품질 향상을 위한 기본 전처리.

        흐름: Grayscale -> Blur -> Adaptive Threshold.
        단순 계산용이므로 aggressive 한 연산은 피함.
        """
        print("[처리] 이미지 전처리 중...")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary

    def extract_and_solve(self, image_path):
        """엔드투엔드 처리 (파일 경로 입력).

        1) 이미지 로드 / 전처리 저장
        2) OCR 수행 -> 텍스트 합치기
        3) 패턴 탐지 후 계산/검증 출력
        """
        try:
            print("[시작] 이미지 처리를 시작합니다...")
            
            # 이미지 로드
            img_array = np.fromfile(image_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("[오류] 이미지를 불러올 수 없습니다.")
                return
            
            print(f"[정보] 이미지 크기: {image.shape[1]}x{image.shape[0]}")
            
            # 전처리
            processed = self.preprocess_image(image)
            
            # 전처리 이미지 저장
            current_dir = os.path.dirname(image_path)
            processed_path = os.path.join(current_dir, "processed_simple.png")
            cv2.imwrite(processed_path, processed)
            print(f"[완료] 전처리 이미지 저장완료")
            
            # OCR로 텍스트 추출
            print("[진행] OCR 텍스트 추출 중...")
            results = self.reader.readtext(image, detail=0)
            all_text = ' '.join(results)
            
            print(f"[추출완료] 텍스트: {all_text}")
            
            if not all_text.strip():
                print("[결과] 텍스트를 찾을 수 없습니다.")
                return
            
            # 수식 분석 및 계산
            print("[분석] 수식을 분석합니다...")
            self.analyze_and_solve(all_text)
            
        except Exception as e:
            print(f"[오류] 처리 중 오류: {str(e)}")

    def analyze_and_solve(self, text):
        """추출된 텍스트에 대해 규칙 기반 분석 & 결과 출력."""
        print("\n" + "="*50)
        print("[결과] 문제 분석 결과")
        print("="*50)
        
        found_something = False
        
        # 등식 찾기
        equations = re.findall(r'\d+\s*[\+\-\*\/×÷]\s*\d+\s*\=\s*\d+', text)
        if equations:
            print("\n등식 검증:")
            for eq in equations:
                self.verify_equation(eq)
                found_something = True
        
        # 연산 찾기
        operations = re.findall(r'\d+\s*[\+\-\*\/×÷]\s*\d+', text)
        if operations:
            print("\n연산 계산:")
            for op in operations:
                if not any(op in eq for eq in equations):  # 등식의 일부가 아닌 경우
                    self.calculate_operation(op)
                    found_something = True
        
        # 좌표 찾기
        coordinates = re.findall(r'[A-Z]?\s*\(\s*[-]?\d+\s*,\s*[-]?\d+\s*\)', text)
        if coordinates:
            print("\n좌표 정보:")
            coord_points = []
            for coord in coordinates:
                point = self.extract_coordinate(coord)
                if point:
                    coord_points.append(point)
                    print(f"   {coord} -> ({point[0]}, {point[1]})")
                    found_something = True
            
            if len(coord_points) >= 2:
                print("\n거리 계산:")
                for i in range(len(coord_points)-1):
                    for j in range(i+1, len(coord_points)):
                        dist = self.calculate_distance(coord_points[i], coord_points[j])
                        print(f"   거리: {dist:.2f}")
        
        # 숫자들
        if not found_something:
            numbers = re.findall(r'\b\d+\b', text)
            if numbers:
                print("\n발견된 숫자들:")
                unique_numbers = list(set(map(int, numbers)))
                for num in unique_numbers:
                    print(f"   {num}")
                
                if len(unique_numbers) >= 2:
                    print(f"\n   합계: {sum(unique_numbers)}")
                    print(f"   곱: {np.prod(unique_numbers)}")
                found_something = True
        
        if not found_something:
            print("   분석 가능한 수식이 없습니다.")
        
        print("="*50)

    def verify_equation(self, equation):
        """등식 좌/우 계산 후 오차 허용 범위 내 동일 여부 판단."""
        try:
            eq = equation.replace('×', '*').replace('÷', '/')
            left, right = eq.split('=')
            
            left_val = eval(left.strip())
            right_val = eval(right.strip())
            
            is_correct = abs(left_val - right_val) < 0.0001
            status = "맞음" if is_correct else "틀림"
            
            print(f"   {equation} -> {status}")
            if not is_correct:
                print(f"      정답: {left_val}")
                
        except:
            print(f"   {equation} -> 계산불가")

    def calculate_operation(self, operation):
        """단순 이항/복합 연산 수행."""
        try:
            op = operation.replace('×', '*').replace('÷', '/')
            result = eval(op)
            print(f"   {operation} = {result}")
        except:
            print(f"   {operation} -> 계산불가")

    def extract_coordinate(self, coord_text):
        """'(x,y)' 패턴 정규식 추출 -> (int,int)."""
        try:
            match = re.search(r'\(\s*([-]?\d+)\s*,\s*([-]?\d+)\s*\)', coord_text)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        except:
            pass
        return None

    def calculate_distance(self, point1, point2):
        """유클리드 거리 sqrt((dx)^2 + (dy)^2)."""
        x1, y1 = point1
        x2, y2 = point2
        return ((x2-x1)**2 + (y2-y1)**2)**0.5

    def run(self):
        """예제 이미지 파일 기반 단일 실행 편의 함수."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "example.png")
        
        print(f"이미지: {image_path}")
        print("="*50)
        
        if not os.path.exists(image_path):
            print(f"이미지 파일 없음: {image_path}")
            return
        
        # 한 번만 실행
        self.extract_and_solve(image_path)
        print("\n처리 완료!")

if __name__ == "__main__":
    solver = SimpleMathSolver()
    solver.run()