"""
Line Tracker - Vision Module
IRC 2026 Humanoid Robot Competition

라인의 중심점들을 원에 피팅하여 곡률과 방향(직선/좌회전/우회전)을 판별합니다.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Direction(Enum):
    STRAIGHT = "직선"
    LEFT = "좌회전"
    RIGHT = "우회전"
    UNKNOWN = "판별불가"


@dataclass
class TrackState:
    direction: Direction
    curvature: float          # 1/radius (클수록 급커브)
    radius: float             # 피팅 원의 반지름 (px), inf면 직선
    center_offset: float      # 로봇 중심선으로부터 라인 중심까지의 오프셋 (px, 음수=왼쪽)
    circle_center: Optional[tuple]  # 피팅 원의 중심 좌표 (x, y)
    confidence: float         # 신뢰도 0~1
    centroids: list           # 검출된 라인 중심점들


# ─────────────────────────────────────────────
#  1. 전처리 + 중심점 추출
# ─────────────────────────────────────────────

def preprocess(frame: np.ndarray, cfg: dict) -> np.ndarray:
    """
    흰색 라인 마커 검출을 위한 전처리.
    HSV 기반 흰색 마스크 → 모폴로지 정리
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(cfg.get("white_lower", [0, 0, 180]))
    upper = np.array(cfg.get("white_upper", [180, 60, 255]))
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def extract_centroids(mask: np.ndarray, cfg: dict) -> list[tuple[float, float]]:
    """
    마스크에서 연결 성분(각 마커 조각)의 중심점 추출.
    너무 작거나 큰 blob은 제거.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    min_area = cfg.get("min_blob_area", 300)
    max_area = cfg.get("max_blob_area", 20000)

    points = []
    for i in range(1, n_labels):  # 0은 배경
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy)))

    # y좌표(세로) 기준으로 정렬 – 아래쪽(가까운 쪽)이 먼저
    points.sort(key=lambda p: -p[1])
    return points


# ─────────────────────────────────────────────
#  2. 원 피팅 (최소자승법)
# ─────────────────────────────────────────────

def fit_circle_lstsq(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """
    최소자승법으로 점들에 원을 피팅.
    반환: (cx, cy, radius) 또는 None (실패 시)

    수식:  x² + y² = 2·cx·x + 2·cy·y + (r² - cx² - cy²)
    선형화: A·[cx, cy, c]ᵀ = b  where c = cx²+cy²-r²
    """
    if len(points) < 3:
        return None

    pts = np.array(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2

    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx, cy = result[0], result[1]
    c = result[2]
    r2 = cx ** 2 + cy ** 2 + c  # c = r² - cx² - cy²  →  r² = c + cx² + cy²
    if r2 <= 0:
        return None

    return float(cx), float(cy), float(np.sqrt(r2))


def fit_quality(points: list[tuple[float, float]],
                cx: float, cy: float, r: float) -> float:
    """
    잔차 기반 신뢰도 계산 (0~1).
    모든 점이 원 위에 가까울수록 1에 가까움.
    """
    pts = np.array(points, dtype=np.float64)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    residuals = np.abs(dists - r)
    mean_res = np.mean(residuals)
    # 평균 잔차가 0이면 1, r의 20% 이상이면 0
    quality = max(0.0, 1.0 - mean_res / (0.2 * r + 1e-6))
    return float(quality)


# ─────────────────────────────────────────────
#  3. 방향 판별
# ─────────────────────────────────────────────

def determine_direction(
    centroids: list[tuple[float, float]],
    circle_result: tuple | None,
    frame_w: int,
    cfg: dict
) -> TrackState:
    """
    피팅 원의 위치를 이용해 방향을 결정.

    핵심 논리:
      - 직선 판별: 피팅 원의 반지름 > straight_threshold
      - 원의 중심이 로봇 기준선(이미지 중앙) 기준으로
          왼쪽  → 오른쪽 커브 (로봇이 오른쪽으로 틀어야 함)
          오른쪽 → 왼쪽 커브
    """
    straight_r  = cfg.get("straight_radius_threshold", 1500)  # px
    min_conf    = cfg.get("min_confidence", 0.3)
    robot_center_x = frame_w / 2.0

    if not centroids:
        return TrackState(
            direction=Direction.UNKNOWN, curvature=0.0, radius=float("inf"),
            center_offset=0.0, circle_center=None, confidence=0.0, centroids=[]
        )

    # 라인 중심 오프셋 (가장 가까운 마커 기준)
    nearest_cx = centroids[0][0]
    center_offset = nearest_cx - robot_center_x

    if circle_result is None:
        # 점이 2개 이하 등 피팅 실패 → 직선으로 fallback
        return TrackState(
            direction=Direction.STRAIGHT, curvature=0.0, radius=float("inf"),
            center_offset=center_offset, circle_center=None, confidence=0.0,
            centroids=centroids
        )

    cx, cy, r = circle_result
    confidence = fit_quality(centroids, cx, cy, r)

    if confidence < min_conf:
        direction = Direction.UNKNOWN
    elif r > straight_r:
        direction = Direction.STRAIGHT
    else:
        # 원의 중심이 로봇 기준선 기준으로 어느 쪽인지
        if cx < robot_center_x:
            direction = Direction.RIGHT   # 원이 왼쪽 → 오른쪽으로 도는 커브
        else:
            direction = Direction.LEFT    # 원이 오른쪽 → 왼쪽으로 도는 커브

    curvature = 1.0 / r if r > 0 else 0.0

    return TrackState(
        direction=direction,
        curvature=curvature,
        radius=r,
        center_offset=center_offset,
        circle_center=(cx, cy),
        confidence=confidence,
        centroids=centroids,
    )


# ─────────────────────────────────────────────
#  4. 통합 파이프라인
# ─────────────────────────────────────────────

def analyze_frame(frame: np.ndarray, cfg: dict) -> tuple[TrackState, np.ndarray]:
    """
    프레임 하나를 받아 TrackState와 시각화 프레임을 반환.
    """
    h, w = frame.shape[:2]

    # ROI: 하단 절반만 처리 (가까운 라인만)
    roi_ratio = cfg.get("roi_top_ratio", 0.4)
    roi_top = int(h * roi_ratio)
    roi = frame[roi_top:, :]

    mask = preprocess(roi, cfg)
    centroids_roi = extract_centroids(mask, cfg)

    # ROI 좌표 → 전체 프레임 좌표로 변환
    centroids = [(cx, cy + roi_top) for cx, cy in centroids_roi]

    circle_result = fit_circle_lstsq(centroids) if len(centroids) >= 3 else None
    state = determine_direction(centroids, circle_result, w, cfg)

    vis = visualize(frame.copy(), mask, state, roi_top, cfg)
    return state, vis


# ─────────────────────────────────────────────
#  5. 시각화
# ─────────────────────────────────────────────

COLOR = {
    Direction.STRAIGHT: (0, 220, 0),
    Direction.LEFT:     (255, 100, 0),
    Direction.RIGHT:    (0, 100, 255),
    Direction.UNKNOWN:  (100, 100, 100),
}


def visualize(frame: np.ndarray, mask: np.ndarray,
              state: TrackState, roi_top: int, cfg: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    color = COLOR[state.direction]

    # ROI 경계선
    cv2.line(frame, (0, roi_top), (w, roi_top), (80, 80, 80), 1)

    # 마스크 오버레이 (반투명)
    mask_color = np.zeros_like(frame)
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[roi_top:] = mask
    mask_color[mask_full > 0] = [255, 255, 0]
    frame = cv2.addWeighted(frame, 1.0, mask_color, 0.35, 0)

    # 중심점 표시
    for cx, cy in state.centroids:
        cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)
        cv2.circle(frame, (int(cx), int(cy)), 7, (255, 255, 255), 1)

    # 피팅 원 표시
    if state.circle_center and state.radius < 3000:
        ccx, ccy = int(state.circle_center[0]), int(state.circle_center[1])
        r = int(state.radius)
        cv2.circle(frame, (ccx, ccy), min(r, 2000), color, 2)
        cv2.drawMarker(frame, (ccx, ccy), color,
                       cv2.MARKER_CROSS, 20, 2)

    # 로봇 중심선
    cv2.line(frame, (w // 2, h), (w // 2, roi_top), (200, 200, 200), 1, cv2.LINE_AA)

    # 오프셋 표시 바
    bar_y = h - 20
    cv2.line(frame, (0, bar_y), (w, bar_y), (60, 60, 60), 18)
    offset_x = int(np.clip(w // 2 + state.center_offset, 0, w))
    cv2.circle(frame, (offset_x, bar_y), 8, color, -1)
    cv2.circle(frame, (w // 2, bar_y), 4, (255, 255, 255), -1)

    # 방향 아이콘 + 텍스트
    arrow_map = {
        Direction.STRAIGHT: "▲ ",
        Direction.LEFT:     "◀ ",
        Direction.RIGHT:    "▶ ",
        Direction.UNKNOWN:  "? ",
    }
    label = arrow_map[state.direction] + state.direction.value
    cv2.rectangle(frame, (8, 8), (340, 130), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (340, 130), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (18, 50), font, 1.1, color, 2, cv2.LINE_AA)

    r_text = f"R: {'inf' if state.radius > 3000 else f'{state.radius:.0f}px'}"
    cv2.putText(frame, r_text, (18, 80), font, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    conf_text = f"Conf: {state.confidence:.2f}  Offset: {state.center_offset:+.0f}px"
    cv2.putText(frame, conf_text, (18, 108), font, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    return frame
