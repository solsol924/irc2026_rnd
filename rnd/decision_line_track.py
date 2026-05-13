# decision_algorithm.py
# 알고리즘 팀 전담 파일: 상태 코드 및 조향각 결정 로직

import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# 1. 방향 열거형
class Direction(Enum):
    STRAIGHT = "직선"
    LEFT     = "좌회전"
    RIGHT    = "우회전"
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

# 3. 신뢰도 계산 (판단 로직에 필요함)
def poly_confidence(points, coeffs):
    if len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float64)
    ys, xs = pts[:, 1], pts[:, 0]
    a, b, c = coeffs
    mean_res = np.abs(xs - (a * ys**2 + b * ys + c)).mean()
    return float(max(0.0, 1.0 - mean_res / 50.0))

# 4. 방향 및 조향각 판별 함수 (★알고리즘 팀이 수정할 핵심 영역★)
def determine_direction(centroids, band_pts, coeffs, frame_w, cfg) -> TrackState:
    robot_cx = frame_w / 2.0
    
    # 선을 잃어버렸을 때 (Status: 5)
    if not centroids:
        return TrackState(Direction.UNKNOWN, 0.0, 0.0, None, 0.0, [], [], status_code=5, steering_angle=0.0)

    center_offset = centroids[0][0] - robot_cx

    if coeffs is None:
        return TrackState(Direction.STRAIGHT, 0.0, center_offset, None, 0.15, centroids, band_pts, status_code=1, steering_angle=0.0)

    a, b, c = coeffs
    confidence = poly_confidence(band_pts if band_pts else centroids, coeffs)

    # --- 튜닝 파라미터 (알고리즘 팀이 조절할 값들) ---
    DELTA_OUT = 120.0        # 복귀 마지노선
    VERTICAL_ANGLE = 15.0    # 직진 인정 각도
    K_CURVE = 20000.0        # 조향각 가중치
    # ----------------------------------------------

    status_code = 1
    direction = Direction.STRAIGHT
    steering_angle = 0.0

    # 복귀 모드 (13, 14 번)
    if abs(center_offset) > DELTA_OUT:
        direction = Direction.UNKNOWN
        if center_offset < -DELTA_OUT:
            status_code = 13
            steering_angle = -45.0  
        else:
            status_code = 14
            steering_angle = 45.0   
    # 정상 주행 모드 (1, 2, 3 번)
    else:
        current_angle = math.degrees(math.atan(b))
        lookahead_angle = a * K_CURVE
        steering_angle = current_angle + lookahead_angle

        if abs(steering_angle) <= VERTICAL_ANGLE:
            status_code = 1
            direction = Direction.STRAIGHT
        elif steering_angle > VERTICAL_ANGLE:
            status_code = 3
            direction = Direction.RIGHT
        elif steering_angle < -VERTICAL_ANGLE:
            status_code = 2
            direction = Direction.LEFT

    return TrackState(
        direction=direction, 
        curvature=float(a), 
        center_offset=center_offset, 
        poly_coeffs=coeffs,
        confidence=confidence, 
        centroids=centroids, 
        band_centroids=band_pts,
        status_code=status_code,         
        steering_angle=steering_angle    
    )