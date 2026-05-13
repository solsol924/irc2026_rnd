# decision_line_track.py
# 알고리즘 팀 전담 파일: 상태 코드 및 조향각 결정 로직

import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# 1. 방향 열거형 (미세 회전 추가)
class Direction(Enum):
    STRAIGHT = "직선"
    LEFT     = "좌회전"
    RIGHT    = "우회전"
    MICRO_LEFT = "미세 좌회전"
    MICRO_RIGHT = "미세 우회전"
    UNKNOWN  = "판별불가"

# 2. 로봇 상태 데이터 클래스
@dataclass
class TrackState:
    direction:      Direction
    curvature:      float                   
    center_offset:  float                   
    poly_coeffs:    Optional[np.ndarray]    
    confidence:     float
    centroids:      list = field(default_factory=list)
    band_centroids: list = field(default_factory=list)
    status_code:    int = 1                 
    steering_angle: float = 0.0             

# 3. 신뢰도 계산
def poly_confidence(points, coeffs):
    if len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float64)
    ys, xs = pts[:, 1], pts[:, 0]
    a, b, c = coeffs
    mean_res = np.abs(xs - (a * ys**2 + b * ys + c)).mean()
    return float(max(0.0, 1.0 - mean_res / 50.0))

# =====================================================================
# ★ 이탈/복귀 상태를 기억하기 위한 메모리 클래스 (Global State) ★
# =====================================================================
class TrackingMemory:
    last_steer_dir = Direction.STRAIGHT  # 직전 조향 방향 (이탈 시 회전 방향 결정용)
    is_out_mode = False                  # 현재 이탈(Out) 모드인지 여부

state_memory = TrackingMemory()

# =====================================================================
# 4. 방향 및 조향각 판별 함수 (핵심 알고리즘)
# =====================================================================
def determine_direction(centroids, band_pts, coeffs, frame_w, cfg) -> TrackState:
    robot_cx = frame_w / 2.0
    robot_cy = cfg.get("cam_height", 480) # 로봇 중심의 y좌표 (화면 맨 아래)
    
    # --- 튜닝 파라미터 (임계값) ---
    DELTA_OUT = 140.0        # 이탈 판단: 중심에서 이 픽셀 이상 벗어나면 OUT
    IN_THRESHOLD = 60.0      # 복귀 완료: 타겟과의 x축 거리가 이 픽셀 이하가 되면 정상 궤도 진입
    A_THRESH = 0.0003        # 곡률 임계값: 이 값 이하면 직선, 이상이면 곡선으로 판단
                             # ※ fit_poly2()에서 y축 간격 부족 시 a=0.0으로 1차 강등되어 넘어오므로,
                             #    그 케이스도 여기서 자동으로 '직선' 분기로 처리됨
    
    ANGLE_MICRO = 10.0       # 미세 회전 기준 (10도 이하 직진)
    ANGLE_TURN = 30.0        # 일반 회전 기준 (30도 이상 좌/우회전)
    # -----------------------------

    # 초기값 세팅
    status_code = 1
    direction = Direction.STRAIGHT
    steering_angle = 0.0
    
    # [조건 1] 선이 안 보이거나 (0점), 로봇과의 거리가 임계 픽셀 이상일 때 이탈(OUT)로 판별
    is_line_lost = (len(centroids) == 0)
    center_offset = (centroids[0][0] - robot_cx) if not is_line_lost else 0.0
    
    if is_line_lost or abs(center_offset) > DELTA_OUT:
        state_memory.is_out_mode = True

    # ==========================================
    # [분기 A] 이탈(Out) 모드 가동 중일 때
    # ==========================================
    if state_memory.is_out_mode:
        if is_line_lost:
            # 라인이 안 보임 -> 직전 프레임의 조향 상태를 기반으로 제자리 회전하며 탐색
            if state_memory.last_steer_dir in [Direction.LEFT, Direction.MICRO_LEFT]:
                status_code = 2 # 좌회전하며 탐색
                direction = Direction.LEFT
                steering_angle = -45.0
            elif state_memory.last_steer_dir in [Direction.RIGHT, Direction.MICRO_RIGHT]:
                status_code = 3 # 우회전하며 탐색
                direction = Direction.RIGHT
                steering_angle = 45.0
            else:
                status_code = 2 # 기본 좌회전 탐색
                direction = Direction.LEFT
                steering_angle = -45.0
                
            return TrackState(direction, 0.0, 0.0, coeffs, 0.0, centroids, band_pts, status_code, steering_angle)
        
        else:
            # 회전하다가 라인이 판별된 순간! 가장 상단(멀리 있는) 점을 타겟으로 잡음
            target_x, target_y = centroids[-1]
            
            # 타겟 좌표를 향해 직진하기 위한 조향각 계산
            dx = target_x - robot_cx
            dy = robot_cy - target_y
            steering_angle = math.degrees(math.atan2(dx, dy))
            
            # 좌표와 로봇과의 X축 거리 측정
            if abs(dx) < IN_THRESHOLD:
                # 거리가 일정 이하가 됨 -> 이탈 모드 종료 및 트래킹 복귀!
                state_memory.is_out_mode = False
            else:
                # 아직 거리가 멈 -> '좌표 따라 걷기(Status 8)' 유지
                status_code = 8
                direction = Direction.STRAIGHT 
                return TrackState(direction, 0.0, center_offset, coeffs, 0.5, centroids, band_pts, status_code, steering_angle)

    # ==========================================
    # [분기 B] 정상 라인 트래킹 모드
    # ==========================================
    if len(centroids) == 2:
        # 점 2개: 이어진 직선의 각도 계산
        p_bot = centroids[0]
        p_top = centroids[1]
        dx = p_top[0] - p_bot[0]
        dy = p_bot[1] - p_top[1] # 영상에서 위로 가는 것이 전방(+)
        steering_angle = math.degrees(math.atan2(dx, dy))
        
    elif len(centroids) >= 3 and coeffs is not None:
        a, b, c = coeffs
        if abs(a) < A_THRESH:
            # a값이 일정 이하 -> '직선' 판단 (가장 먼 점과 가까운 점을 이음)
            p_bot = centroids[0]
            p_top = centroids[-1]
            dx = p_top[0] - p_bot[0]
            dy = p_bot[1] - p_top[1]
            steering_angle = math.degrees(math.atan2(dx, dy))
        else:
            # a값이 일정 이상 -> '곡선' 판단
            # ★ 핵심: 가장 상단 점에서의 접선을 그어 각도 계산
            y_top = centroids[-1][1]
            
            # x = ay^2 + by + c 방정식의 미분: dx/dy = 2ay + b
            # 영상 좌표계 특성상 로봇 전방(위쪽)으로 갈 때 y축 변화량(dy)은 -1
            dx_dt = -(2 * a * y_top + b)
            dy_dt = 1.0 
            steering_angle = math.degrees(math.atan2(dx_dt, dy_dt))
            
    # 각도(steering_angle)에 따른 상태(Status) 세분화
    abs_angle = abs(steering_angle)
    
    if abs_angle <= ANGLE_MICRO:
        status_code = 1    # 10도 이하: 직진
        direction = Direction.STRAIGHT
        
    elif ANGLE_MICRO < abs_angle <= ANGLE_TURN:
        if steering_angle < 0:
            status_code = 4 # 10~30도: 미세 좌회전
            direction = Direction.MICRO_LEFT
        else:
            status_code = 5 # 10~30도: 미세 우회전
            direction = Direction.MICRO_RIGHT
            
    else: # 30도 초과
        if steering_angle < 0:
            status_code = 2 # 30도 이상: 좌회전
            direction = Direction.LEFT
        else:
            status_code = 3 # 30도 이상: 우회전
            direction = Direction.RIGHT

    # 정상 트래킹 중일 때, 직전 조향 상태를 변수에 저장해 둠 (나중에 이탈 시 써먹기 위해)
    if status_code in [2, 3, 4, 5]:
        state_memory.last_steer_dir = direction

    # a값 추출 및 안전 처리
    curvature_val = float(coeffs[0]) if coeffs is not None else 0.0
    confidence = poly_confidence(band_pts if band_pts else centroids, coeffs) if coeffs is not None else 0.5

    # ★ 이 마지막 return 부분이 제일 중요합니다! 들여쓰기가 앞으로 딱 붙어있어야 에러가 안 납니다.
    return TrackState(
        direction=direction, 
        curvature=curvature_val, 
        center_offset=center_offset, 
        poly_coeffs=coeffs,
        confidence=confidence, 
        centroids=centroids, 
        band_centroids=band_pts,
        status_code=status_code,         
        steering_angle=steering_angle    
    )