"""math_solver.py

여러 수학 분야(기하, 대수, 해석, 확률·통계, 위상, 응용, 함수, 사칙연산)를
OCR (EasyOCR / Tesseract)로 추출한 텍스트를 기반으로 간단 규칙/휴리스틱으로
분석·출력하는 데모/프로토타입 통합 솔버.

구조 개요:
    - GeometrySolver: 좌표·도형 분석 (Shapely 있으면 고급, 없으면 기본 계산)
    - AlgebraSolver: 방정식/다항식 패턴 탐지와 간단 풀이
    - AnalysisSolver: 극한/미분/적분/급수 키워드 탐지 (설명 위주)
    - ProbabilitySolver: 확률 관련 키워드, 조합/순열 계산
    - TopologySolver: 위상 용어 및 오일러 특성수 간단 추론
    - AppliedMathSolver: 최적화/미분방정식/푸리에 등 키워드 안내
    - FunctionSolver: 함수식 패턴 탐지 및 일부 값 평가
    - ArithmeticSolver: 사칙 연산/등식 검증
    - MathSolver: 위 솔버들을 통합하여 이미지 또는 텍스트 입력 처리

주의:
    - 본 코드는 교육/프로토타이핑 목적. 정밀한 수학적 검증/파서는 구현하지 않음.
    - 정규식 기반 단순 탐지이므로 오검출/누락 발생 가능.
    - eval 사용 부분(간단 계산)은 외부 입력에 대해 보안 리스크가 있으므로
        실제 서비스 전에는 안전한 파서로 교체 필요.
"""

import os
import cv2
import numpy as np
import re
import math
from typing import List, Tuple, Dict, Any
from collections import Counter
import itertools

# OCR 라이브러리들
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Shapely 라이브러리
try:
    from shapely.geometry import Point, LineString, Polygon
    from shapely.ops import cascaded_union
    import shapely.affinity as affinity
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

class GeometrySolver:
        """기하학 문제 풀이 (Shapely 기반 선택적).

        Shapely 설치 여부에 따라:
            - 설치됨: Polygon, LineString 등 활용한 정확 계산
            - 미설치: 기본 거리/넓이/분류 수식 수동 계산
        """

        def __init__(self):
                self.use_shapely = SHAPELY_AVAILABLE
                if not self.use_shapely:
                        print("[경고] Shapely가 없어서 기본 계산을 사용합니다.")

        @staticmethod
        def find_coordinates(text: str) -> Tuple[List[Tuple[float, float]], List[str]]:
        """좌표 추출 (소수점 지원)"""
        coord_patterns = [
            r'[A-Z]\s*\(\s*([-]?\d+(?:\.\d+)?)\s*,\s*([-]?\d+(?:\.\d+)?)\s*\)',  # A(3.5,4.2)
            r'\(\s*([-]?\d+(?:\.\d+)?)\s*,\s*([-]?\d+(?:\.\d+)?)\s*\)'           # (3.5,4.2)
        ]
        
        coordinates = []
        coord_names = []
        
        for pattern in coord_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    coordinates.append((x, y))
                    
                    # 점 이름 추출
                    full_match = match.group(0)
                    name_match = re.match(r'([A-Z])', full_match)
                    if name_match:
                        coord_names.append(name_match.group(1))
                    else:
                        coord_names.append(f"점{len(coordinates)}")
                except:
                    continue
        
        return coordinates, coord_names
    
    def create_shapely_points(self, coordinates: List[Tuple[float, float]]) -> List[Point]:
        """Shapely Point 객체들 생성"""
        if not self.use_shapely:
            return []
        return [Point(x, y) for x, y in coordinates]
    
    def analyze_triangle(self, points: List[Tuple[float, float]], names: List[str]) -> Dict[str, Any]:
        """삼각형 분석.

        반환 항목: 변 길이, 각, 둘레, 넓이, 무게중심, 외심/내심 및 반지름 등.
        Shapely 가능 시 polygon.length/area 활용, 아니면 기본 공식 사용.
        """
        if len(points) != 3:
            return {}
        
        if self.use_shapely:
            # Shapely를 사용한 고급 분석
            shapely_points = self.create_shapely_points(points)
            triangle = Polygon(points)
            
            # 변 생성
            line_ab = LineString([points[0], points[1]])
            line_bc = LineString([points[1], points[2]]) 
            line_ca = LineString([points[2], points[0]])
            
            # 변의 길이
            side_ab = line_ab.length
            side_bc = line_bc.length
            side_ca = line_ca.length
            
            # 넓이와 둘레
            area = triangle.area
            perimeter = triangle.length
            
            # 무게중심
            centroid = triangle.centroid
            
            # 외접원
            circumcenter = self.calculate_circumcenter(points)
            circumradius = self.calculate_circumradius(points)
            
        else:
            # 기본 계산
            a, b, c = points
            side_ab = math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
            side_bc = math.sqrt((c[0]-b[0])**2 + (c[1]-b[1])**2)
            side_ca = math.sqrt((a[0]-c[0])**2 + (a[1]-c[1])**2)
            
            perimeter = side_ab + side_bc + side_ca
            area = abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])) / 2)
            centroid = ((a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3)
            circumcenter = self.calculate_circumcenter(points)
            circumradius = self.calculate_circumradius(points)
        
        # 삼각형 종류 판별
        sides = sorted([side_ab, side_bc, side_ca])
        triangle_type = self.classify_triangle(sides)
        
        # 각도 계산
        angles = self.calculate_triangle_angles(points)
        
        return {
            'sides': [side_ab, side_bc, side_ca],
            'side_names': [f"{names[0]}{names[1]}", f"{names[1]}{names[2]}", f"{names[2]}{names[0]}"],
            'angles': angles,
            'perimeter': perimeter,
            'area': area,
            'type': triangle_type,
            'centroid': centroid,
            'circumcenter': circumcenter,
            'circumradius': circumradius,
            'incenter': self.calculate_incenter(points),
            'inradius': area / (perimeter / 2) if perimeter > 0 else 0
        }
    
    def analyze_polygon(self, points: List[Tuple[float, float]], names: List[str]) -> Dict[str, Any]:
        """다각형 분석 (Shapely 기반)"""
        if len(points) < 3:
            return {}
        
        if self.use_shapely:
            polygon = Polygon(points)
            
            # 기본 속성
            area = polygon.area
            perimeter = polygon.length
            centroid = polygon.centroid
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            
            # 볼록한 다각형인지 확인
            convex_hull = polygon.convex_hull
            is_convex = polygon.equals(convex_hull)
            
            # 단순 다각형인지 확인 (자기 교차 없음)
            is_simple = polygon.is_valid and polygon.is_simple
            
        else:
            # 기본 계산
            area = self.calculate_polygon_area(points)
            perimeter = self.calculate_polygon_perimeter(points)
            centroid = self.calculate_polygon_centroid(points)
            is_convex = self.is_convex_polygon(points)
            is_simple = True  # 간단히 참으로 가정
        
        # 변의 길이들
        sides = []
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            if self.use_shapely:
                line = LineString([points[i], points[next_i]])
                sides.append(line.length)
            else:
                p1, p2 = points[i], points[next_i]
                length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                sides.append(length)
        
        # 다각형 종류 판별
        polygon_type = self.classify_polygon(len(points), sides, is_convex)
        
        return {
            'vertices': len(points),
            'sides': sides,
            'area': area,
            'perimeter': perimeter,
            'centroid': centroid,
            'type': polygon_type,
            'is_convex': is_convex,
            'is_simple': is_simple
        }
    
    def calculate_circumcenter(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """외심 계산"""
        if len(points) != 3:
            return (0, 0)
        
        (x1, y1), (x2, y2), (x3, y3) = points
        
        # 외심 공식
        d = 2 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        if abs(d) < 1e-10:
            return (0, 0)  # 일직선상의 점들
        
        ux = ((x1**2 + y1**2)*(y2-y3) + (x2**2 + y2**2)*(y3-y1) + (x3**2 + y3**2)*(y1-y2)) / d
        uy = ((x1**2 + y1**2)*(x3-x2) + (x2**2 + y2**2)*(x1-x3) + (x3**2 + y3**2)*(x2-x1)) / d
        
        return (ux, uy)
    
    def calculate_circumradius(self, points: List[Tuple[float, float]]) -> float:
        """외접원 반지름 계산"""
        if len(points) != 3:
            return 0
        
        circumcenter = self.calculate_circumcenter(points)
        if circumcenter == (0, 0):
            return 0
        
        # 외심에서 한 점까지의 거리
        p1 = points[0]
        return math.sqrt((circumcenter[0] - p1[0])**2 + (circumcenter[1] - p1[1])**2)
    
    def calculate_incenter(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """내심 계산"""
        if len(points) != 3:
            return (0, 0)
        
        (x1, y1), (x2, y2), (x3, y3) = points
        
        # 변의 길이
        a = math.sqrt((x2-x3)**2 + (y2-y3)**2)  # BC
        b = math.sqrt((x1-x3)**2 + (y1-y3)**2)  # AC  
        c = math.sqrt((x1-x2)**2 + (y1-y2)**2)  # AB
        
        if a + b + c == 0:
            return (0, 0)
        
        # 내심 공식
        ix = (a*x1 + b*x2 + c*x3) / (a + b + c)
        iy = (a*y1 + b*y2 + c*y3) / (a + b + c)
        
        return (ix, iy)
    
    def calculate_triangle_angles(self, points: List[Tuple[float, float]]) -> List[float]:
        """삼각형 내각 계산 (도 단위)"""
        if len(points) != 3:
            return []
        
        # 변의 길이
        a = math.sqrt((points[1][0]-points[2][0])**2 + (points[1][1]-points[2][1])**2)
        b = math.sqrt((points[0][0]-points[2][0])**2 + (points[0][1]-points[2][1])**2)
        c = math.sqrt((points[0][0]-points[1][0])**2 + (points[0][1]-points[1][1])**2)
        
        # 코사인 법칙으로 각도 계산
        try:
            angle_A = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / math.pi
            angle_B = math.acos((a**2 + c**2 - b**2) / (2*a*c)) * 180 / math.pi  
            angle_C = 180 - angle_A - angle_B
            return [angle_A, angle_B, angle_C]
        except:
            return [0, 0, 0]
    
    def classify_triangle(self, sides: List[float]) -> str:
        """삼각형 종류 판별"""
        sides = sorted(sides)
        a, b, c = sides
        
        types = []
        
        # 변의 길이에 따른 분류
        if abs(a - b) < 0.001 and abs(b - c) < 0.001:
            types.append("정삼각형")
        elif abs(a - b) < 0.001 or abs(b - c) < 0.001 or abs(a - c) < 0.001:
            types.append("이등변삼각형")
        else:
            types.append("부등변삼각형")
        
        # 각도에 따른 분류
        if abs(a**2 + b**2 - c**2) < 0.001:
            types.append("직각삼각형")
        elif a**2 + b**2 < c**2:
            types.append("둔각삼각형")
        else:
            types.append("예각삼각형")
        
        return " & ".join(types)
    
    def classify_polygon(self, vertices: int, sides: List[float], is_convex: bool) -> str:
        """다각형 종류 판별"""
        base_names = {
            3: "삼각형", 4: "사각형", 5: "오각형", 
            6: "육각형", 7: "칠각형", 8: "팔각형"
        }
        
        base_name = base_names.get(vertices, f"{vertices}각형")
        
        # 정다각형 확인
        if len(set(f"{s:.3f}" for s in sides)) == 1:  # 모든 변이 같음
            base_name = f"정{base_name}"
        
        # 볼록/오목 분류
        if not is_convex:
            base_name = f"오목{base_name}"
        
        return base_name
    
    def calculate_polygon_area(self, points: List[Tuple[float, float]]) -> float:
        """다각형 넓이 (신발끈 공식)"""
        n = len(points)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2
    
    def calculate_polygon_perimeter(self, points: List[Tuple[float, float]]) -> float:
        """다각형 둘레"""
        perimeter = 0
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            p1, p2 = points[i], points[next_i]
            perimeter += math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        return perimeter
    
    def calculate_polygon_centroid(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """다각형 무게중심"""
        n = len(points)
        cx = sum(p[0] for p in points) / n
        cy = sum(p[1] for p in points) / n
        return (cx, cy)
    
    def is_convex_polygon(self, points: List[Tuple[float, float]]) -> bool:
        """볼록 다각형인지 확인"""
        n = len(points)
        if n < 3:
            return False
        
        sign = None
        for i in range(n):
            p1 = points[i]
            p2 = points[(i+1) % n]
            p3 = points[(i+2) % n]
            
            # 외적 계산
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            if abs(cross) > 1e-10:
                if sign is None:
                    sign = cross > 0
                elif (cross > 0) != sign:
                    return False
        
        return True
    
    def solve(self, text: str) -> None:
        """기하 문제 풀이 진입점.

        1) 좌표 정규식으로 추출
        2) 모든 점 쌍 거리
        3) 점 개수에 따라 삼각형/다각형 추가 분석
        """
        print(f"\n🔷 기하학 문제 분석 {'(Shapely 기반)' if self.use_shapely else '(기본 계산)'}")
        print("-" * 50)
        
        coordinates, coord_names = self.find_coordinates(text)
        
        if not coordinates:
            print("좌표를 찾을 수 없습니다.")
            return
        
        print(f"발견된 좌표:")
        for name, coord in zip(coord_names, coordinates):
            print(f"   {name}: {coord}")
        
        # 기본 거리 계산
        if len(coordinates) >= 2:
            print(f"\n📏 거리 계산:")
            for i in range(len(coordinates)):
                for j in range(i+1, len(coordinates)):
                    if self.use_shapely:
                        p1, p2 = Point(coordinates[i]), Point(coordinates[j])
                        dist = p1.distance(p2)
                    else:
                        p1, p2 = coordinates[i], coordinates[j]
                        dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    
                    print(f"   {coord_names[i]} ↔ {coord_names[j]} = {dist:.3f}")
        
        # 도형별 상세 분석
        if len(coordinates) == 3:
            result = self.analyze_triangle(coordinates, coord_names)
            if result:
                print(f"\n🔺 삼각형 상세 분석:")
                print(f"   종류: {result['type']}")
                print(f"   변의 길이: {[f'{s:.3f}' for s in result['sides']]}")
                print(f"   내각: {[f'{a:.1f}°' for a in result['angles']]}")
                print(f"   둘레: {result['perimeter']:.3f}")
                print(f"   넓이: {result['area']:.3f}")
                print(f"   무게중심: ({result['centroid'][0]:.3f}, {result['centroid'][1]:.3f})")
                print(f"   외심: ({result['circumcenter'][0]:.3f}, {result['circumcenter'][1]:.3f})")
                print(f"   외접원 반지름: {result['circumradius']:.3f}")
                print(f"   내심: ({result['incenter'][0]:.3f}, {result['incenter'][1]:.3f})")
                print(f"   내접원 반지름: {result['inradius']:.3f}")
        
        elif len(coordinates) >= 4:
            result = self.analyze_polygon(coordinates, coord_names)
            if result:
                print(f"\n🔷 다각형 상세 분석:")
                print(f"   종류: {result['type']}")
                print(f"   꼭짓점 개수: {result['vertices']}")
                print(f"   변의 길이: {[f'{s:.3f}' for s in result['sides']]}")
                print(f"   둘레: {result['perimeter']:.3f}")
                print(f"   넓이: {result['area']:.3f}")
                print(f"   무게중심: ({result['centroid'][0]:.3f}, {result['centroid'][1]:.3f})")
                print(f"   볼록 다각형: {'예' if result['is_convex'] else '아니오'}")
                print(f"   단순 다각형: {'예' if result['is_simple'] else '아니오'}")

class AlgebraSolver:
    """대수학 문제 풀이.

    단순 정규식 기반 패턴으로 일/이차 방정식 후보 추출 -> 기본 공식 사용.
    정교한 파싱/부호 처리/다항식 전개 등은 생략된 축약형 구현.
    """
    
    @staticmethod
    def find_equations(text: str) -> List[str]:
        """방정식 찾기 - 확장"""
        patterns = [
            r'[a-z]\s*[\+\-]?\s*\d+\s*=\s*\d+',  # x + 5 = 10
            r'\d+\s*[a-z]\s*[\+\-]?\s*\d+\s*=\s*\d+',  # 2x + 3 = 7
            r'[a-z]\s*=\s*\d+',  # x = 5
            r'\d+\s*[a-z]\s*=\s*\d+',  # 2x = 10
            r'[a-z]\s*\^\s*\d+',  # 이차식 감지용
            r'[a-z]\^2\s*[\+\-]\s*\d*[a-z]?\s*[\+\-]?\s*\d*\s*=\s*\d+',  # ax^2 + bx + c = 0
        ]
        
        equations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            equations.extend(matches)
        
        return equations
    
    @staticmethod
    def find_polynomials(text: str) -> List[str]:
        """다항식 찾기"""
        patterns = [
            r'[a-z]\^2\s*[\+\-]?\s*\d*[a-z]?\s*[\+\-]?\s*\d+',  # ax^2 + bx + c
            r'\d*[a-z]\^3\s*[\+\-]?\s*\d*[a-z]\^2\s*[\+\-]?\s*\d*[a-z]?\s*[\+\-]?\s*\d+',  # 삼차다항식
        ]
        
        polynomials = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            polynomials.extend(matches)
        
        return polynomials
    
    @staticmethod
    def solve_quadratic(equation: str) -> Dict[str, Any]:
        """이차방정식 풀이"""
        try:
            # ax^2 + bx + c = 0 형태로 변환
            if '=' in equation:
                left, right = equation.split('=')
                # 간단한 이차방정식 파싱 (예: x^2 + 2x + 1 = 0)
                # 실제로는 더 복잡한 파싱이 필요하지만 간단히 구현
                
                # 계수 추출 (매우 기본적인 구현)
                a, b, c = 1, 0, 0
                if 'x^2' in left:
                    # a 계수 찾기
                    coeff_match = re.search(r'([-]?\d*)\s*x\^2', left)
                    if coeff_match:
                        coeff_str = coeff_match.group(1)
                        if coeff_str in ['', '+']:
                            a = 1
                        elif coeff_str == '-':
                            a = -1
                        else:
                            a = float(coeff_str)
                
                # 판별식 계산
                discriminant = b**2 - 4*a*c
                
                if discriminant >= 0:
                    x1 = (-b + math.sqrt(discriminant)) / (2*a)
                    x2 = (-b - math.sqrt(discriminant)) / (2*a)
                    return {
                        'type': 'quadratic',
                        'coefficients': [a, b, c],
                        'discriminant': discriminant,
                        'solutions': [x1, x2] if discriminant > 0 else [x1],
                        'original': equation
                    }
                else:
                    return {
                        'type': 'quadratic',
                        'coefficients': [a, b, c],
                        'discriminant': discriminant,
                        'solutions': [],
                        'note': '실근이 없습니다',
                        'original': equation
                    }
        except:
            pass
        
        return {}
    
    @staticmethod
    def solve_linear_equation(equation: str) -> Dict[str, Any]:
        """일차방정식 풀이"""
        try:
            # 변수 찾기
            variable_match = re.search(r'([a-z])', equation)
            if not variable_match:
                return {}
            
            variable = variable_match.group(1)
            
            # 등호 기준으로 나누기
            left, right = equation.split('=')
            left = left.strip()
            right = right.strip()
            
            # 우변 값
            right_val = float(right)
            
            # 좌변에서 계수와 상수 추출
            coeff_pattern = r'([-]?\d*)\s*' + variable + r'\s*([\+\-]?\s*\d+)?'
            match = re.match(coeff_pattern, left)
            
            if not match:
                return {}
            
            coeff_str = match.group(1)
            const_str = match.group(2)
            
            # 계수 처리
            if coeff_str == '' or coeff_str == '+':
                coeff = 1
            elif coeff_str == '-':
                coeff = -1
            else:
                coeff = float(coeff_str)
            
            # 상수 처리
            const = 0
            if const_str:
                const_str = const_str.replace(' ', '')
                const = float(const_str)
            
            # 해 구하기: ax + b = c => x = (c - b) / a
            if coeff != 0:
                solution = (right_val - const) / coeff
                return {
                    'type': 'linear',
                    'variable': variable,
                    'solution': solution,
                    'original': equation
                }
            
        except:
            pass
        
        return {}
    
    def solve(self, text: str) -> None:
        """대수 문제 풀이.

        1) 방정식 패턴 수집
        2) 이차/일차 식별 후 간단 해 계산 (가능한 범위)
        3) 다항식은 향후 확장 포인트로 안내 출력
        """
        print("\n🔶 대수학 문제 분석")
        print("-" * 30)
        
        equations = self.find_equations(text)
        polynomials = self.find_polynomials(text)
        
        if not equations and not polynomials:
            print("방정식이나 다항식을 찾을 수 없습니다.")
            return
        
        if equations:
            print("발견된 방정식:")
            for eq in equations:
                print(f"   {eq}")
                
                # 이차방정식 확인
                if '^2' in eq or 'x²' in eq:
                    result = self.solve_quadratic(eq)
                    if result:
                        print(f"   → 이차방정식 해: {result.get('solutions', [])}")
                        if 'note' in result:
                            print(f"      {result['note']}")
                else:
                    # 일차방정식 풀이
                    result = self.solve_linear_equation(eq)
                    if result:
                        print(f"   → {result['variable']} = {result['solution']:.2f}")
        
        if polynomials:
            print("\n발견된 다항식:")
            for poly in polynomials:
                print(f"   {poly}")
                print("   → 다항식 분석 (인수분해, 근 등)")

class AnalysisSolver:
    """해석학 문제 풀이.

    극한/미분/적분/급수 키워드만 탐지 후 개념적 힌트 출력.
    구체적 계산 엔진 없음 (sympy 등 연계 가능).
    """
    
    @staticmethod
    def find_limits(text: str) -> List[str]:
        """극한 찾기"""
        patterns = [
            r'lim\s*[a-z]?\s*→\s*\d+',  # lim x→0
            r'limit\s*[a-z]?\s*→\s*\d+',
            r'극한',
        ]
        
        limits = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            limits.extend(matches)
        
        return limits
    
    @staticmethod
    def find_derivatives(text: str) -> List[str]:
        """도함수/미분 찾기"""
        patterns = [
            r"f'\s*\([a-z]\)",  # f'(x)
            r"d[a-z]/d[a-z]",   # dy/dx
            r"미분",
            r"도함수",
        ]
        
        derivatives = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            derivatives.extend(matches)
        
        return derivatives
    
    @staticmethod
    def find_integrals(text: str) -> List[str]:
        """적분 찾기"""
        patterns = [
            r'∫\s*[^d]*d[a-z]',  # ∫f(x)dx
            r'integral',
            r'적분',
            r'부정적분',
            r'정적분',
        ]
        
        integrals = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            integrals.extend(matches)
        
        return integrals
    
    @staticmethod
    def find_series(text: str) -> List[str]:
        """급수 찾기"""
        patterns = [
            r'∑',  # 시그마 기호
            r'급수',
            r'수열',
            r'series',
            r'sequence',
        ]
        
        series = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            series.extend(matches)
        
        return series
    
    @staticmethod
    def calculate_basic_derivative(function_str: str) -> str:
        """기본 도함수 계산"""
        try:
            # 매우 기본적인 도함수 규칙들
            if 'x^2' in function_str:
                return "2x"
            elif 'x^3' in function_str:
                return "3x^2"
            elif function_str.strip() == 'x':
                return "1"
            elif function_str.isdigit():
                return "0"
            else:
                return f"d/dx({function_str})"
        except:
            return "계산 불가"
    
    def solve(self, text: str) -> None:
        """해석학 문제 풀이"""
        print("\n🔸 해석학 문제 분석")
        print("-" * 30)
        
        limits = self.find_limits(text)
        derivatives = self.find_derivatives(text)
        integrals = self.find_integrals(text)
        series = self.find_series(text)
        
        if not any([limits, derivatives, integrals, series]):
            print("해석학 관련 내용을 찾을 수 없습니다.")
            return
        
        if limits:
            print("🎯 극한 분석:")
            for limit in limits:
                print(f"   {limit}")
                print("   → 극한값 계산 (L'Hôpital 정리 적용 가능)")
        
        if derivatives:
            print("\n📈 미분 분석:")
            for deriv in derivatives:
                print(f"   {deriv}")
                print("   → 연쇄법칙, 곱의법칙, 몫의법칙 적용")
        
        if integrals:
            print("\n📊 적분 분석:")
            for integral in integrals:
                print(f"   {integral}")
                print("   → 치환적분, 부분적분, 부분분수 적용")
        
        if series:
            print("\n🔢 급수 분석:")
            for ser in series:
                print(f"   {ser}")
                print("   → 수렴성 판정 (비 판정법, 근 판정법 등)")

class ProbabilitySolver:
    """확률·통계 문제 풀이.

    용어 감지 + 간단 조합/순열 + 대표적 확률 사전 지식 출력.
    대규모 샘플링/분포 계산은 미구현.
    """
    
    @staticmethod
    def find_probability_terms(text: str) -> List[str]:
        """확률 관련 용어 찾기"""
        terms = [
            '확률', 'probability', 'P(', 
            '주사위', 'dice', '동전', 'coin',
            '카드', 'card', '뽑기', 'draw',
            '평균', 'mean', '분산', 'variance',
            '표준편차', 'standard deviation',
            '정규분포', 'normal distribution',
            '이항분포', 'binomial distribution'
        ]
        
        found_terms = []
        for term in terms:
            if term.lower() in text.lower():
                found_terms.append(term)
        
        return found_terms
    
    @staticmethod
    def calculate_combination(n: int, r: int) -> int:
        """조합 계산 nCr"""
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        
        # nCr = n! / (r! * (n-r)!)
        numerator = 1
        denominator = 1
        
        for i in range(r):
            numerator *= (n - i)
            denominator *= (i + 1)
        
        return numerator // denominator
    
    @staticmethod
    def calculate_permutation(n: int, r: int) -> int:
        """순열 계산 nPr"""
        if r > n or r < 0:
            return 0
        if r == 0:
            return 1
        
        result = 1
        for i in range(n, n - r, -1):
            result *= i
        
        return result
    
    @staticmethod
    def find_numbers_in_text(text: str) -> List[int]:
        """텍스트에서 숫자 찾기"""
        numbers = re.findall(r'\d+', text)
        return [int(n) for n in numbers]
    
    def solve(self, text: str) -> None:
        """확률·통계 문제 풀이"""
        print("\n🎲 확률·통계 문제 분석")
        print("-" * 30)
        
        prob_terms = self.find_probability_terms(text)
        numbers = self.find_numbers_in_text(text)
        
        if not prob_terms:
            print("확률·통계 관련 내용을 찾을 수 없습니다.")
            return
        
        print(f"발견된 확률·통계 용어: {prob_terms}")
        print(f"발견된 숫자: {numbers}")
        
        # 조합/순열 계산 시도
        if len(numbers) >= 2:
            n, r = numbers[0], numbers[1]
            if n >= r and n <= 20:  # 계산 가능한 범위
                comb = self.calculate_combination(n, r)
                perm = self.calculate_permutation(n, r)
                print(f"\n🔢 조합·순열 계산:")
                print(f"   C({n}, {r}) = {comb}")
                print(f"   P({n}, {r}) = {perm}")
        
        # 기본 확률 계산
        if '주사위' in text or 'dice' in text.lower():
            print("\n🎯 주사위 확률 분석:")
            print("   - 한 개 주사위: 각 면이 나올 확률 = 1/6")
            print("   - 두 개 주사위: 총 경우의 수 = 36")
        
        if '동전' in text or 'coin' in text.lower():
            print("\n🪙 동전 확률 분석:")
            print("   - 앞면 또는 뒷면: 각각 1/2")
            if len(numbers) > 0:
                n = numbers[0]
                if n <= 10:
                    print(f"   - {n}번 던지기: 총 경우의 수 = 2^{n} = {2**n}")

class TopologySolver:
    """위상수학 문제 풀이.

    오일러 특성수 χ = V - E + F 기본 공식 및 키워드 기반 안내.
    실제 위상 동형성 판정 알고리즘은 포함하지 않음.
    """
    
    @staticmethod
    def find_topology_terms(text: str) -> List[str]:
        """위상수학 관련 용어 찾기"""
        terms = [
            '위상', 'topology', '연결', 'connected',
            '컴팩트', 'compact', '열린집합', 'open set',
            '닫힌집합', 'closed set', '근방', 'neighborhood',
            '연속', 'continuous', '동형사상', 'homeomorphism',
            '오일러 특성수', 'euler characteristic',
            '기본군', 'fundamental group'
        ]
        
        found_terms = []
        for term in terms:
            if term.lower() in text.lower():
                found_terms.append(term)
        
        return found_terms
    
    @staticmethod
    def calculate_euler_characteristic(vertices: int, edges: int, faces: int) -> int:
        """오일러 특성수 계산 V - E + F"""
        return vertices - edges + faces
    
    @staticmethod
    def classify_surface(euler_char: int) -> str:
        """오일러 특성수로 곡면 분류"""
        if euler_char == 2:
            return "구면 (Sphere)"
        elif euler_char == 1:
            return "사영평면 (Projective Plane)"
        elif euler_char == 0:
            return "토러스 (Torus) 또는 클라인 병 (Klein Bottle)"
        elif euler_char < 0:
            genus = (2 - euler_char) // 2
            return f"종수 {genus}인 곡면"
        else:
            return "알 수 없는 곡면"
    
    def solve(self, text: str) -> None:
        """위상수학 문제 풀이"""
        print("\n🌐 위상수학 문제 분석")
        print("-" * 30)
        
        topo_terms = self.find_topology_terms(text)
        
        if not topo_terms:
            print("위상수학 관련 내용을 찾을 수 없습니다.")
            return
        
        print(f"발견된 위상수학 용어: {topo_terms}")
        
        # 오일러 특성수 관련 분석
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 3:
            v, e, f = int(numbers[0]), int(numbers[1]), int(numbers[2])
            euler_char = self.calculate_euler_characteristic(v, e, f)
            surface_type = self.classify_surface(euler_char)
            
            print(f"\n📐 오일러 특성수 분석:")
            print(f"   꼭짓점(V): {v}, 모서리(E): {e}, 면(F): {f}")
            print(f"   오일러 특성수: χ = V - E + F = {euler_char}")
            print(f"   곡면 종류: {surface_type}")
        
        # 기본적인 위상 개념들
        if '연속' in text or 'continuous' in text.lower():
            print("\n🔄 연속성 분석:")
            print("   - 함수의 연속성 확인")
            print("   - 위상적 성질 보존")
        
        if '동형사상' in text or 'homeomorphism' in text.lower():
            print("\n🔗 동형사상 분석:")
            print("   - 위상적으로 같은 도형 판별")
            print("   - 불변량 계산")

class AppliedMathSolver:
    """응용수학 문제 풀이.

    최적화/미분방정식/푸리에 변환 등 고급 분야 키워드 감지 -> 학습 가이드 메시지.
    실제 해법(심플렉스, ODE solver 등)은 추후 연계 가능.
    """
    
    @staticmethod
    def find_applied_terms(text: str) -> List[str]:
        """응용수학 관련 용어 찾기"""
        terms = [
            '최적화', 'optimization', '선형계획법', 'linear programming',
            '미분방정식', 'differential equation', 'ODE', 'PDE',
            '푸리에', 'fourier', '변환', 'transform',
            '신호처리', 'signal processing', '제어', 'control',
            '게임이론', 'game theory', '경제', 'economics',
            '물리', 'physics', '공학', 'engineering'
        ]
        
        found_terms = []
        for term in terms:
            if term.lower() in text.lower():
                found_terms.append(term)
        
        return found_terms
    
    @staticmethod
    def solve_linear_programming_2d(c1: float, c2: float, constraints: List[Tuple]) -> Dict:
        """2차원 선형계획법 간단한 해법"""
        # 매우 기본적인 구현 - 실제로는 심플렉스 방법 등이 필요
        print("   → 선형계획법 해법 적용 필요")
        print("   → 최적해는 제약조건의 꼭짓점에서 발생")
        return {"method": "graphical_method", "note": "그래프 해법 적용"}
    
    @staticmethod
    def identify_differential_equation(equation: str) -> str:
        """미분방정식 종류 식별"""
        if "y''" in equation or "d²y/dx²" in equation:
            return "2차 미분방정식"
        elif "y'" in equation or "dy/dx" in equation:
            return "1차 미분방정식"
        elif "∂" in equation:
            return "편미분방정식 (PDE)"
        else:
            return "미분방정식 형태 불명확"
    
    def solve(self, text: str) -> None:
        """응용수학 문제 풀이"""
        print("\n🔧 응용수학 문제 분석")
        print("-" * 30)
        
        applied_terms = self.find_applied_terms(text)
        
        if not applied_terms:
            print("응용수학 관련 내용을 찾을 수 없습니다.")
            return
        
        print(f"발견된 응용수학 용어: {applied_terms}")
        
        # 최적화 문제 분석
        if any(term in applied_terms for term in ['최적화', 'optimization', '선형계획법']):
            print("\n📈 최적화 문제 분석:")
            print("   - 목적함수와 제약조건 식별")
            print("   - 라그랑주 승수법 또는 선형계획법 적용")
            
            # 숫자가 있으면 간단한 분석
            numbers = re.findall(r'\d+', text)
            if len(numbers) >= 2:
                print(f"   - 발견된 계수: {numbers}")
        
        # 미분방정식 분석
        if any(term in applied_terms for term in ['미분방정식', 'differential', 'ODE', 'PDE']):
            print("\n🧮 미분방정식 분석:")
            equations = re.findall(r"[^.!?]*[dy'/dx|y''|∂][^.!?]*", text)
            for eq in equations:
                eq_type = self.identify_differential_equation(eq)
                print(f"   {eq.strip()} → {eq_type}")
        
        # 푸리에 변환 분석
        if any(term in applied_terms for term in ['푸리에', 'fourier', '변환']):
            print("\n🌊 푸리에 분석:")
            print("   - 주파수 도메인 변환")
            print("   - 신호의 주파수 성분 분석")

class FunctionSolver:
    """함수 문제 풀이.

    f(x)= ... 형태 탐지 후 간단 분류 및 여러 테스트 점에서 평가.
    안전 문제(eval) 존재 -> 운영 시 safe parser로 교체 권장.
    """
    
    @staticmethod
    def find_functions(text: str) -> List[str]:
        """함수 찾기"""
        patterns = [
            r'f\s*\(\s*x\s*\)\s*=\s*[^=\n]+',  # f(x) = ...
            r'y\s*=\s*[^=\n]+',  # y = ...
            r'[a-z]\s*\(\s*[x-z]\s*\)\s*=\s*[^=\n]+',  # g(x) = ...
            r'[a-z]\s*=\s*[^=\n]+[a-z][^=\n]*',  # 함수 형태
        ]
        
        functions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            functions.extend(matches)
        
        return functions
    
    @staticmethod
    def classify_function(func_str: str) -> str:
        """함수 분류"""
        if '²' in func_str or '^2' in func_str:
            if '³' in func_str or '^3' in func_str:
                return "3차 이상 다항함수"
            else:
                return "2차 함수 (포물선)"
        elif 'sin' in func_str or 'cos' in func_str or 'tan' in func_str:
            return "삼각함수"
        elif 'log' in func_str or 'ln' in func_str:
            return "로그함수"
        elif 'e^' in func_str or '지수' in func_str:
            return "지수함수"
        elif '/' in func_str and any(var in func_str for var in 'xyz'):
            return "유리함수"
        else:
            return "일차함수 또는 기타"
    
    @staticmethod
    def evaluate_function(func_str: str, x_val: float) -> float:
        """함수값 계산"""
        try:
            # f(x) = 부분 제거
            if '=' in func_str:
                func_str = func_str.split('=')[1].strip()
            
            # x를 실제 값으로 치환
            func_str = func_str.replace('x', str(x_val))
            func_str = func_str.replace('^', '**')  # 거듭제곱 변환
            
            return eval(func_str)
        except:
            return None
    
    def solve(self, text: str) -> None:
        """함수 문제 풀이"""
        print("\n🔸 함수 문제 분석")
        print("-" * 30)
        
        functions = self.find_functions(text)
        
        if not functions:
            print("함수를 찾을 수 없습니다.")
            return
        
        print("발견된 함수:")
        for func in functions:
            print(f"   {func}")
            
            # 함수 분류
            func_type = self.classify_function(func)
            print(f"   → 함수 종류: {func_type}")
            
            # 몇 가지 x 값에 대해 함수값 계산
            test_values = [0, 1, -1, 2, -2]
            print("   함수값:")
            for x in test_values:
                y = self.evaluate_function(func, x)
                if y is not None:
                    print(f"     x={x} → y={y}")
            
            # 함수의 성질 분석
            print("   → 정의역, 치역, 단조성, 대칭성 분석 필요")

class ArithmeticSolver:
    """사칙연산/등식 검증.

    정규식 기반 식 추출 -> eval 로 계산/검증. (보안주의)
    """
    
    @staticmethod
    def find_operations(text: str) -> List[str]:
        """연산식 찾기"""
        patterns = [
            r'\d+\s*[\+\-\*\/×÷]\s*\d+\s*=\s*\d+',  # 등식
            r'\d+\s*[\+\-\*\/×÷]\s*\d+(?!\s*=)',    # 연산식
            r'\d+\s*[\+\-\*\/×÷]\s*\d+\s*[\+\-\*\/×÷]\s*\d+',  # 복합연산
        ]
        
        operations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            operations.extend(matches)
        
        return operations
    
    @staticmethod
    def calculate_operation(operation: str) -> float:
        """연산 계산"""
        try:
            op = operation.replace('×', '*').replace('÷', '/')
            if '=' in op:
                op = op.split('=')[0].strip()
            return eval(op)
        except:
            return None
    
    @staticmethod
    def verify_equation(equation: str) -> bool:
        """등식 검증"""
        try:
            eq = equation.replace('×', '*').replace('÷', '/')
            left, right = eq.split('=')
            left_val = eval(left.strip())
            right_val = eval(right.strip())
            return abs(left_val - right_val) < 0.0001
        except:
            return False
    
    def solve(self, text: str) -> None:
        """사칙연산 문제 풀이"""
        print("\n🔹 사칙연산 문제 분석")
        print("-" * 30)
        
        operations = self.find_operations(text)
        
        if not operations:
            print("연산식을 찾을 수 없습니다.")
            return
        
        print("발견된 연산:")
        for op in operations:
            if '=' in op:
                # 등식 검증
                is_correct = self.verify_equation(op)
                status = "✓ 맞음" if is_correct else "✗ 틀림"
                print(f"   {op} → {status}")
                
                if not is_correct:
                    correct_answer = self.calculate_operation(op)
                    if correct_answer is not None:
                        print(f"      정답: {correct_answer}")
            else:
                # 연산 계산
                result = self.calculate_operation(op)
                if result is not None:
                    print(f"   {op} = {result}")

class MathSolver:
    def __init__(self):
        print("🚀 종합 수학 문제 풀이 시스템 (6개 분야 통합)")
        print("="*70)
        
        # OCR 초기화
        self.ocr_method = None
        if EASYOCR_AVAILABLE:
            try:
                print("[설정] EasyOCR 초기화 중...")
                self.reader = easyocr.Reader(['ko', 'en'])
                self.ocr_method = 'easyocr'
                print("[설정] ✅ EasyOCR 초기화 완료")
            except:
                pass
        
        if not self.ocr_method and PYTESSERACT_AVAILABLE:
            try:
                print("[설정] Tesseract 초기화 중...")
                pytesseract.get_tesseract_version()
                self.ocr_method = 'tesseract'
                print("[설정] ✅ Tesseract 초기화 완료")
            except:
                pass
        
        if not self.ocr_method:
            raise Exception("❌ OCR 라이브러리를 찾을 수 없습니다.")
        
        # Shapely 상태 확인
        if SHAPELY_AVAILABLE:
            print("[설정] ✅ Shapely 사용 가능 - 고급 기하 분석 활성화")
        else:
            print("[설정] ⚠️  Shapely 없음 - 기본 기하 계산 사용")
        
        # 6개 분야 솔버들 초기화
        self.geometry_solver = GeometrySolver()        # 기하학
        self.algebra_solver = AlgebraSolver()          # 대수학
        self.analysis_solver = AnalysisSolver()        # 해석학 (새로 추가)
        self.probability_solver = ProbabilitySolver()   # 확률·통계 (새로 추가)
        self.topology_solver = TopologySolver()        # 위상수학 (새로 추가)
        self.applied_solver = AppliedMathSolver()       # 응용수학 (새로 추가)
        self.function_solver = FunctionSolver()        # 함수
        self.arithmetic_solver = ArithmeticSolver()    # 사칙연산
        
        print("-" * 70)
        print("📚 지원하는 수학 분야:")
        print("   🔷 기하학 (Geometry) - 점, 선, 도형, 삼각형, 다각형")
        print("   🔶 대수학 (Algebra) - 방정식, 다항식, 인수분해")
        print("   🔸 해석학 (Analysis) - 미적분, 극한, 급수")
        print("   🎲 확률·통계 (Probability) - 확률, 조합, 통계")
        print("   🌐 위상수학 (Topology) - 연속성, 오일러 특성수")
        print("   🔧 응용수학 (Applied) - 최적화, 미분방정식")
        print("   🔸 함수 (Functions) - 다양한 함수의 성질")
        print("   🔹 사칙연산 (Arithmetic) - 기본 계산")
        print("-" * 70)

    def extract_text(self, image):
        """이미지에서 텍스트 추출"""
        if self.ocr_method == 'easyocr':
            results = self.reader.readtext(image, detail=False)
            return ' '.join(results)
        
        elif self.ocr_method == 'tesseract':
            custom_config = r'--oem 3 --psm 6 -l kor+eng'
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
        
        return ""

    def preprocess_image(self, image):
        """이미지 전처리"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 이진화
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary

    def analyze_problem_type(self, text: str) -> List[str]:
        """문제 유형 분석 - 6개 분야 확장"""
        problem_types = []
        
        # 1. 기하학 (Geometry)
        geometry_keywords = ['좌표', '점', '삼각형', '사각형', '원', '둘레', '넓이', 'triangle', 'circle']
        if (re.search(r'\(\s*[-]?\d+\s*,\s*[-]?\d+\s*\)', text) or 
            any(keyword in text.lower() for keyword in geometry_keywords)):
            problem_types.append('geometry')
        
        # 2. 대수학 (Algebra)
        algebra_keywords = ['방정식', 'equation', '다항식', 'polynomial']
        if (re.search(r'[a-z]\s*[\+\-\*\/=]', text) or
            any(keyword in text.lower() for keyword in algebra_keywords)):
            problem_types.append('algebra')
        
        # 3. 해석학 (Analysis)
        analysis_keywords = ['극한', 'limit', '미분', 'derivative', '적분', 'integral', '급수', 'series']
        if any(keyword in text.lower() for keyword in analysis_keywords):
            problem_types.append('analysis')
        
        # 4. 확률·통계 (Probability)
        prob_keywords = ['확률', 'probability', '주사위', 'dice', '동전', 'coin', '조합', 'combination']
        if any(keyword in text.lower() for keyword in prob_keywords):
            problem_types.append('probability')
        
        # 5. 위상수학 (Topology)
        topo_keywords = ['위상', 'topology', '연속', 'continuous', '오일러', 'euler']
        if any(keyword in text.lower() for keyword in topo_keywords):
            problem_types.append('topology')
        
        # 6. 응용수학 (Applied)
        applied_keywords = ['최적화', 'optimization', '미분방정식', 'differential', '푸리에', 'fourier']
        if any(keyword in text.lower() for keyword in applied_keywords):
            problem_types.append('applied')
        
        # 7. 함수 (Function)
        if (re.search(r'[a-z]\s*\(\s*[x-z]\s*\)\s*=', text) or 
            re.search(r'f\s*\(\s*x\s*\)', text) or 'function' in text.lower()):
            problem_types.append('function')
        
        # 8. 사칙연산 (Arithmetic)
        if re.search(r'\d+\s*[\+\-\*\/×÷]\s*\d+', text):
            problem_types.append('arithmetic')
        
        return problem_types if problem_types else ['arithmetic']

    def solve(self, image_path):
        """문제 풀이 메인 함수"""
        try:
            print(f"📸 이미지 분석: {image_path}")
            print("="*60)
            
            # 이미지 로드
            img_array = np.fromfile(image_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("[오류] 이미지를 불러올 수 없습니다.")
                return
            
            # 전처리
            processed = self.preprocess_image(image)
            
            # OCR로 텍스트 추출
            print(f"[진행] {self.ocr_method.upper()}로 텍스트 추출...")
            text = self.extract_text(image)
            
            print(f"[추출된 텍스트] {text}")
            
            if not text.strip():
                print("[결과] 텍스트를 찾을 수 없습니다.")
                return
            
            # 문제 유형 분석
            problem_types = self.analyze_problem_type(text)
            print(f"\n[분석] 🎯 감지된 문제 유형: {', '.join(problem_types)}")
            
            # 각 유형별 솔버 실행
            for ptype in problem_types:
                if ptype == 'geometry':
                    self.geometry_solver.solve(text)
                elif ptype == 'algebra':
                    self.algebra_solver.solve(text)
                elif ptype == 'analysis':
                    self.analysis_solver.solve(text)
                elif ptype == 'probability':
                    self.probability_solver.solve(text)
                elif ptype == 'topology':
                    self.topology_solver.solve(text)
                elif ptype == 'applied':
                    self.applied_solver.solve(text)
                elif ptype == 'function':
                    self.function_solver.solve(text)
                elif ptype == 'arithmetic':
                    self.arithmetic_solver.solve(text)
            
            print(f"\n{'='*60}")
            print("✅ 종합 수학 문제 분석 완료!")
            print(f"📊 분석된 분야: {len(problem_types)}개")
            print(f"🧮 처리된 솔버: {', '.join(problem_types)}")
            print("💡 더 정확한 분석을 위해서는 수식과 그림을 명확히 해주세요.")
            
        except Exception as e:
            print(f"[오류] {str(e)}")

    def run(self):
        """실행"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "example.png")
        
        if not os.path.exists(image_path):
            print(f"이미지 파일이 없습니다: {image_path}")
            print("example.png 파일을 현재 디렉토리에 놓고 다시 실행해주세요.")
            return
        
        self.solve(image_path)
    
    def solve_text(self, text: str):
        """텍스트로 직접 문제 풀이 (테스트용)"""
        print("📝 텍스트 기반 수학 문제 분석")
        print("="*60)
        print(f"[입력 텍스트] {text}")
        
        # 문제 유형 분석
        problem_types = self.analyze_problem_type(text)
        print(f"\n[분석] 🎯 감지된 문제 유형: {', '.join(problem_types)}")
        
        # 각 유형별 솔버 실행
        for ptype in problem_types:
            if ptype == 'geometry':
                self.geometry_solver.solve(text)
            elif ptype == 'algebra':
                self.algebra_solver.solve(text)
            elif ptype == 'analysis':
                self.analysis_solver.solve(text)
            elif ptype == 'probability':
                self.probability_solver.solve(text)
            elif ptype == 'topology':
                self.topology_solver.solve(text)
            elif ptype == 'applied':
                self.applied_solver.solve(text)
            elif ptype == 'function':
                self.function_solver.solve(text)
            elif ptype == 'arithmetic':
                self.arithmetic_solver.solve(text)
        
        print(f"\n{'='*60}")
        print("✅ 텍스트 기반 수학 문제 분석 완료!")


def demo_examples():
    """데모 예제들"""
    print("\n" + "🎮 데모 예제 실행" + "="*50)
    
    solver = MathSolver()
    
    examples = [
        # 기하학 예제
        ("기하학", "삼각형 A(0,0), B(3,0), C(0,4)의 넓이와 둘레를 구하시오"),
        
        # 대수학 예제  
        ("대수학", "2x + 5 = 13 방정식을 풀어라"),
        
        # 해석학 예제
        ("해석학", "f(x) = x^2의 도함수를 구하시오"),
        
        # 확률론 예제
        ("확률론", "주사위 2개를 던질 때 합이 7이 나올 확률은?"),
        
        # 위상수학 예제
        ("위상수학", "꼭짓점 8개, 모서리 12개, 면 6개인 다면체의 오일러 특성수"),
        
        # 응용수학 예제
        ("응용수학", "f(x) = x^2 + 2x + 1의 최솟값을 구하는 최적화 문제"),
        
        # 함수 예제
        ("함수", "f(x) = 2x + 1에서 x=3일 때의 함수값"),
        
        # 사칙연산 예제
        ("사칙연산", "25 + 17 × 3 - 8 ÷ 2 = ?")
    ]
    
    for category, example_text in examples:
        print(f"\n{'🔥 ' + category + ' 예제':-^60}")
        solver.solve_text(example_text)
        print("\n" + "-"*60)


if __name__ == "__main__":
    try:
        # 메인 솔버 실행
        solver = MathSolver()
        
        # 이미지 파일이 있으면 이미지 분석, 없으면 데모 실행
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "example.png")
        
        if os.path.exists(image_path):
            print("🖼️  이미지 파일을 발견했습니다. 이미지 분석을 시작합니다...")
            solver.run()
        else:
            print("📝 이미지 파일이 없습니다. 데모 예제를 실행합니다...")
            demo_examples()
            
        print(f"\n{'🎉 프로그램 종료':-^60}")
        
    except Exception as e:
        print(f"❌ 프로그램 실행 오류: {str(e)}")
        print("필요한 라이브러리를 설치해주세요:")
        print("pip install opencv-python numpy easyocr shapely")