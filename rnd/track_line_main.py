"""
Line Tracker - Vision Module
IRC 2026 Humanoid Robot Competition

2차 다항식 피팅(Polynomial Fitting)으로 곡률/방향 판별.
- 직사각형 마커 조건으로 잡음 제거 (가로/세로 비율 필터)
- 로봇이 다른 방향을 바라볼 때도 강건하게 동작
- ROI를 세로 n_bands 구간으로 나눠 균일 샘플링 후 2차 피팅
- settings.ini 에서 모든 파라미터 로드
"""
import configparser
import cv2
import numpy as np
import time
from pathlib import Path
from rnd.decision_line_track import Direction, TrackState, determine_direction


# ═══════════════════════════════════════════════════════
#  settings.ini 로더
# ═══════════════════════════════════════════════════════

def load_config(ini_path: str = "settings.ini") -> dict:
    """
    settings.ini -> cfg dict 변환.
    파일이 없으면 하드코딩된 기본값 사용.
    """
    defaults = {
        "cam_index": 0, "cam_width": 640, "cam_height": 480,
        "cam_fps": 30, "flip_vertical": False,
        "white_lower": [0, 0, 170], "white_upper": [180, 55, 255],
        "min_blob_area": 400, "max_blob_area": 25000,
        "min_aspect_ratio": 1.4, "max_aspect_ratio": 8.0,
        "roi_top_ratio": 0.10, "roi_bottom_ratio": 1.00,
        "roi_left_ratio": 0.18, "roi_right_ratio": 0.82,
        "n_bands": 3,
        "straight_curvature_thresh": 3e-5,
        "min_confidence": 0.30, "min_points_for_poly": 3,
        "show_window": True, "save_video": "",
        "print_every_n_frames": 5,
    }

    p = Path(ini_path)
    if not p.exists():
        print(f"[WARN] {ini_path} not found -> using defaults")
        return defaults

    ini = configparser.ConfigParser()
    ini.read(p, encoding="utf-8")

    def gi(s, k, fb): return ini.getint(s, k, fallback=fb)
    def gf(s, k, fb): return ini.getfloat(s, k, fallback=fb)
    def gb(s, k, fb): return ini.getboolean(s, k, fallback=fb)
    def gs(s, k, fb): return ini.get(s, k, fallback=fb)

    return {
        "cam_index":    gi("camera", "index",         defaults["cam_index"]),
        "cam_width":    gi("camera", "width",          defaults["cam_width"]),
        "cam_height":   gi("camera", "height",         defaults["cam_height"]),
        "cam_fps":      gi("camera", "fps",            defaults["cam_fps"]),
        "flip_vertical": gb("camera", "flip_vertical", defaults["flip_vertical"]),
        "white_lower": [
            gi("detection", "white_lower_h", defaults["white_lower"][0]),
            gi("detection", "white_lower_s", defaults["white_lower"][1]),
            gi("detection", "white_lower_v", defaults["white_lower"][2]),
        ],
        "white_upper": [
            gi("detection", "white_upper_h", defaults["white_upper"][0]),
            gi("detection", "white_upper_s", defaults["white_upper"][1]),
            gi("detection", "white_upper_v", defaults["white_upper"][2]),
        ],
        "min_blob_area":    gi("detection", "min_blob_area",    defaults["min_blob_area"]),
        "max_blob_area":    gi("detection", "max_blob_area",    defaults["max_blob_area"]),
        "min_aspect_ratio": gf("detection", "min_aspect_ratio", defaults["min_aspect_ratio"]),
        "max_aspect_ratio": gf("detection", "max_aspect_ratio", defaults["max_aspect_ratio"]),
        "roi_top_ratio":    gf("detection", "roi_top_ratio",    defaults["roi_top_ratio"]),
        "roi_bottom_ratio": gf("detection", "roi_bottom_ratio", defaults["roi_bottom_ratio"]),
        "roi_left_ratio":   gf("detection", "roi_left_ratio",   defaults["roi_left_ratio"]),
        "roi_right_ratio":  gf("detection", "roi_right_ratio",  defaults["roi_right_ratio"]),
        "n_bands":          gi("detection", "n_bands",           defaults["n_bands"]),
        "straight_curvature_thresh": gf("curve", "straight_curvature_thresh",
                                        defaults["straight_curvature_thresh"]),
        "min_confidence":            gf("curve", "min_confidence",
                                        defaults["min_confidence"]),
        "min_points_for_poly":       gi("curve", "min_points_for_poly",
                                        defaults["min_points_for_poly"]),
        "show_window":          gb("output", "show_window",          defaults["show_window"]),
        "save_video":           gs("output", "save_video",           defaults["save_video"]),
        "print_every_n_frames": gi("output", "print_every_n_frames", defaults["print_every_n_frames"]),
    }


# ═══════════════════════════════════════════════════════
#  1. 전처리 – HSV 마스크 + 직사각형 마커 필터
# ═══════════════════════════════════════════════════════

def preprocess(frame_roi: np.ndarray, cfg: dict) -> np.ndarray:
    hsv   = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg["white_lower"], dtype=np.uint8)
    upper = np.array(cfg["white_upper"], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    return mask


def _is_rectangular(contour: np.ndarray, cfg: dict) -> bool:
    # 1. 최소 면적 직사각형(Bounding Box) 계산
    rect  = cv2.minAreaRect(contour)
    w, h_ = rect[1]
    
    if w < 1 or h_ < 1:
        return False
        
    # 2. 가로/세로 비율(Aspect Ratio)
    aspect = max(w, h_) / min(w, h_)
    if not (cfg["min_aspect_ratio"] <= aspect <= cfg["max_aspect_ratio"]):
        return False

    # 3. 채움 비율(Extent) - 사각형 바운딩 박스 대비 실제 면적
    contour_area = cv2.contourArea(contour)
    bounding_box_area = w * h_
    if bounding_box_area == 0:
        return False
    extent = contour_area / bounding_box_area
    if extent < 0.70: 
        return False

    # 4. ★ 강화 1: Solidity (고무줄 면적 대비 실제 면적) ★
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    solidity = contour_area / hull_area
    # 완벽한 사각형은 1.0. 찌그러지거나 파인 노이즈는 이 수치가 확 떨어집니다.
    if solidity < 0.85: 
        return False

    # 5. ★ 강화 2: 꼭짓점 개수 확인 (다각형 근사) ★
    # 윤곽선 둘레의 4% 정도 오차를 허용하여 다각형으로 단순화시킵니다.
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 진짜 마커(직사각형)라면 꼭짓점이 4개여야 합니다. (카메라 왜곡을 고려해 4~6개까지 허용)
    num_vertices = len(approx)
    if num_vertices < 4 or num_vertices > 6:
        return False

    return True


def extract_centroids(mask: np.ndarray, cfg: dict,
                      offset_x: int = 0, offset_y: int = 0
                      ) -> list[tuple[float, float]]:
    n_labels, labels, stats, cc = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (cfg["min_blob_area"] <= area <= cfg["max_blob_area"]):
            continue
        blob_mask = (labels == i).astype(np.uint8) * 255
        cnts, _   = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or not _is_rectangular(cnts[0], cfg):
            continue
        points.append((float(cc[i][0]) + offset_x, float(cc[i][1]) + offset_y))
    points.sort(key=lambda p: -p[1])
    return points


# ═══════════════════════════════════════════════════════
#  2. 구간별 대표 중심점 (밴드 샘플링)
# ═══════════════════════════════════════════════════════

def band_sample(centroids, roi_top, roi_bottom, n_bands):
    if not centroids:
        return []
    band_h = (roi_bottom - roi_top) / n_bands
    result = []
    for b in range(n_bands):
        y_lo = roi_top + b * band_h
        y_hi = roi_top + (b + 1) * band_h
        pts  = [(cx, cy) for cx, cy in centroids if y_lo <= cy < y_hi]
        if pts:
            result.append((float(np.median([p[0] for p in pts])),
                           float(np.mean  ([p[1] for p in pts]))))
    return result

# ═══════════════════════════════════════════════════════
#  3. 2차 다항식 피팅  x = a*y^2 + b*y + c (2점일 경우 1차 피팅)
# ═══════════════════════════════════════════════════════

def fit_poly2(points):
    if len(points) < 2:
        return None
        
    pts = np.array(points, dtype=np.float64)
    ys, xs = pts[:, 1], pts[:, 0]
    
    def fallback_to_straight():
        idx_bot = int(np.argmax(ys))
        idx_top = int(np.argmin(ys))
        dy = ys[idx_top] - ys[idx_bot]
        if abs(dy) < 1e-3:
            return np.array([0.0, 0.0, float(xs.mean())])
        b_val = (xs[idx_top] - xs[idx_bot]) / dy
        c_val = xs[idx_bot] - b_val * ys[idx_bot]
        return np.array([0.0, b_val, c_val])

    if len(points) == 2:
        return fallback_to_straight()

    # ★ 방어 1: X좌표 순간이동 검사 (가로로 너무 멀리 뛰면 노이즈)
    MAX_X_JUMP = 120 
    ys_sorted_idx = np.argsort(-ys) 
    for i in range(len(ys_sorted_idx)-1):
        x1 = xs[ys_sorted_idx[i]]
        x2 = xs[ys_sorted_idx[i+1]]
        if abs(x2 - x1) > MAX_X_JUMP:
            return fallback_to_straight()

    # ★ 방어 3 (NEW!): 지그재그(방향성) 검사
    # 밑에서 위로 올라갈 때, x의 변화 방향(부호)이 갑자기 바뀌면 안 됩니다.
    # 즉, 계속 우회전하거나 계속 좌회전해야지, 우회전했다 좌회전하는 건 비정상입니다.
    if len(points) >= 3:
        x_diffs = []
        for i in range(len(ys_sorted_idx)-1):
            dx = xs[ys_sorted_idx[i+1]] - xs[ys_sorted_idx[i]]
            x_diffs.append(dx)
            
        # 첫 번째 점과 두 번째 점의 x 변화량 부호
        sign_1 = np.sign(x_diffs[0]) if abs(x_diffs[0]) > 5 else 0 # 5픽셀 미만 미세 변화는 무시
        # 두 번째 점과 세 번째 점의 x 변화량 부호
        sign_2 = np.sign(x_diffs[1]) if abs(x_diffs[1]) > 5 else 0
        
        # 부호가 명확히 반대일 경우 (지그재그 꺾임 발생)
        if sign_1 * sign_2 < 0:
            return fallback_to_straight() # 갈매기 곡선 금지! 직선으로 강등

    MIN_Y_SPAN = 60  
    MIN_Y_GAP  = 20  
    y_span = float(ys.max() - ys.min())
    ys_sorted = np.sort(ys)
    min_gap   = float(np.diff(ys_sorted).min()) if len(ys_sorted) > 1 else 0.0

    if y_span < MIN_Y_SPAN or min_gap < MIN_Y_GAP:
        return fallback_to_straight()

    y_mean = ys.mean()
    y_std  = ys.std() if ys.std() > 1e-6 else 1.0
    yn     = (ys - y_mean) / y_std
    try:
        a_n, b_n, c_n = np.polyfit(yn, xs, 2)
    except (np.linalg.LinAlgError, ValueError):
        return fallback_to_straight()
        
    s = y_std
    a = a_n / s**2
    b = -2 * a_n * y_mean / s**2 + b_n / s
    c = a_n * y_mean**2 / s**2 - b_n * y_mean / s + c_n
    
    # ★ 방어 2: 최대 곡률 제한 (비정상 급커브 방지)
    MAX_CURVATURE = 0.006
    if abs(a) > MAX_CURVATURE:
        return fallback_to_straight()
        
    return np.array([a, b, c])


# ═══════════════════════════════════════════════════════
#  4. 통합 파이프라인
# ═══════════════════════════════════════════════════════

def analyze_frame(frame: np.ndarray, cfg: dict) -> tuple[TrackState, np.ndarray]:
    h, w = frame.shape[:2]
    roi_top    = int(h * cfg["roi_top_ratio"])
    roi_bottom = int(h * cfg["roi_bottom_ratio"])
    roi_left   = int(w * cfg["roi_left_ratio"])
    roi_right  = int(w * cfg["roi_right_ratio"])

    roi       = frame[roi_top:roi_bottom, roi_left:roi_right]
    mask_roi  = preprocess(roi, cfg)
    
    # 여기서 추출된 점들은 이미 아래쪽(가까운 쪽)부터 위쪽 순서로 정렬되어 있습니다.
    centroids_roi = extract_centroids(mask_roi, cfg, offset_x=roi_left, offset_y=roi_top)
    
    # [징검다리(체인) 로직] - 연속된 점만 묶습니다!
    continuous_pts = []
    if centroids_roi:
        continuous_pts.append(centroids_roi[0])
        MAX_GAP = 160  # 점과 점 사이의 최대 허용 거리
        
        for pt in centroids_roi[1:]:
            prev_pt = continuous_pts[-1]
            dist = ((pt[0] - prev_pt[0])**2 + (pt[1] - prev_pt[1])**2)**0.5
            
            if dist < MAX_GAP:
                continuous_pts.append(pt)
            else:
                break
    
    # ★ 핵심 수정: 낡은 band_sample 기능을 완전히 삭제합니다!
    # 걸러진 연속된 점(징검다리)을 5개까지 그대로 가져와서 실제 트랙의 궤적을 만듭니다.
    target_pts = continuous_pts[:5] 
    
    # 2차 함수 피팅도 가짜 점이 아닌, 진짜 연속된 점(target_pts)으로만 계산합니다.
    coeffs  = fit_poly2(target_pts) if len(target_pts) >= cfg["min_points_for_poly"] else None

    # 상태 판별 시에도 진짜 점 정보만 넘겨줍니다. (band_pts 자리에도 똑같이 target_pts 전달)
    state = determine_direction(target_pts, target_pts, coeffs, w, cfg)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[roi_top:roi_bottom, roi_left:roi_right] = mask_roi

    vis = visualize(frame.copy(), full_mask, state,
                    roi_top, roi_bottom, roi_left, roi_right, w, h)
    return state, vis


# ═══════════════════════════════════════════════════════
#  6. 시각화
# ═══════════════════════════════════════════════════════

COLOR = {
    Direction.STRAIGHT: (0, 220, 0),
    Direction.LEFT:     (255, 100, 0),
    Direction.RIGHT:    (0, 100, 255),
    Direction.MICRO_LEFT: (200, 150, 0),    # ★ 새로 추가!
    Direction.MICRO_RIGHT: (0, 150, 200),   # ★ 새로 추가!
    Direction.UNKNOWN:  (120, 120, 120),
}
ARROW = {
    Direction.STRAIGHT: "UP   STRAIGHT",
    Direction.LEFT:     "LEFT  LEFT-TURN",
    Direction.RIGHT:    "RIGHT RIGHT-TURN",
    Direction.MICRO_LEFT: "uL   MICRO-LEFT",    # ★ 새로 추가!
    Direction.MICRO_RIGHT: "uR   MICRO-RIGHT",  # ★ 새로 추가!
    Direction.UNKNOWN:  "?    UNKNOWN",
}


def visualize(frame, mask, state, roi_top, roi_bottom, roi_left, roi_right, w, h):
    color = COLOR[state.direction]
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (80, 80, 80), 2)

    overlay = np.zeros_like(frame)
    overlay[mask > 0] = [255, 255, 0]
    frame = cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)

    for cx, cy in state.centroids:
        cv2.circle(frame, (int(cx), int(cy)), 4, (200, 200, 200), -1)
    for cx, cy in state.band_centroids:
        cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)
        cv2.circle(frame, (int(cx), int(cy)), 9, (255, 255, 255), 1)

    if state.poly_coeffs is not None:
        a, b, c = state.poly_coeffs
        pts_curve = []
        for y_px in range(roi_top, roi_bottom, 4):
            x_px = int(a * y_px**2 + b * y_px + c)
            if roi_left - 20 <= x_px <= roi_right + 20:
                pts_curve.append((x_px, y_px))
        if len(pts_curve) > 1:
            cv2.polylines(frame,
                          [np.array(pts_curve, dtype=np.int32).reshape(-1, 1, 2)],
                          False, color, 2, cv2.LINE_AA)

    cv2.line(frame, (w // 2, h), (w // 2, roi_top), (200, 200, 200), 1, cv2.LINE_AA)

    bar_y = h - 20
    cv2.line(frame, (0, bar_y), (w, bar_y), (50, 50, 50), 18)
    ox = int(np.clip(w // 2 + state.center_offset, 0, w))
    cv2.circle(frame, (ox, bar_y), 8, color, -1)
    cv2.circle(frame, (w // 2, bar_y), 4, (255, 255, 255), -1)

    cv2.rectangle(frame, (8, 8), (250, 105), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (250, 105), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame, ARROW[state.direction],
                (15, 32), font, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Curv: {state.curvature:.2e}",
                (15, 55), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Conf: {state.confidence:.2f}  Off: {state.center_offset:+.0f}px",
                (15, 75), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Pts: {len(state.centroids)}  Bands: {len(state.band_centroids)}",
                (15, 95), font, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    # === 초록색 접선 및 각도 시각화 ===
    if len(state.centroids) >= 3 and state.poly_coeffs is not None:
        import math
        if state.band_centroids and len(state.band_centroids) >= 2:
            mid = len(state.band_centroids) // 2
            target_cx, target_cy = int(state.band_centroids[mid][0]), int(state.band_centroids[mid][1])
        else:
            target_cx, target_cy = int(state.centroids[1][0]), int(state.centroids[1][1])
        
        # 1. 중심선 (회색 선) - 로봇의 직진 방향이므로 화면 '위쪽(-y)'으로 그립니다!
        cv2.line(frame, (target_cx, target_cy), (target_cx, target_cy - 80), (180, 180, 180), 2, cv2.LINE_AA)
        
        # 2. 순수 기하학적 접선 (초록색 선) 재계산
        a, b, c = state.poly_coeffs
        dx_dy = 2 * a * target_cy + b
        pure_tangent_angle = math.degrees(math.atan(-dx_dy))
        pure_angle_rad = math.radians(pure_tangent_angle)
        
        # 위쪽(-y)으로 뻗어나가며, 각도에 따라 x축으로 휘어짐
        end_x = int(target_cx + 80 * math.sin(pure_angle_rad))
        end_y = int(target_cy - 80 * math.cos(pure_angle_rad)) 
        
        cv2.line(frame, (target_cx, target_cy), (end_x, end_y), (0, 255, 0), 3, cv2.LINE_AA)
        
        # 3. 각도 텍스트 표시 (화면에 그어진 순수한 곡선의 각도)
        cv2.putText(frame, f"Tan: {pure_tangent_angle:.1f} deg", 
                    (target_cx + 20, target_cy - 10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 실제 로봇 바퀴/모터가 꺾어야 하는 최종 각도 (Offset 보정 포함)
        cv2.putText(frame, f"Steer: {state.steering_angle:.1f} deg", 
                    (target_cx + 20, target_cy + 15), font, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    # ==============================================================
    # ==============================================================

# 1. 선을 완전히 잃어버려서 회전하며 탐색 중일 때 (점 0개)
    if len(state.centroids) == 0:
        import time
        # 빨간색 굵은 테두리도 거슬린다면 두께(10 -> 2)를 얇게 하거나 아예 지우셔도 됩니다.
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 2)
        
        # 왼쪽 상단 정보창 바로 아래에 깔끔하게 텍스트만 깜빡이게 표시
        if int(time.time() * 5) % 2 == 0:
            cv2.putText(frame, "SEARCHING...", (15, 125), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "SEARCHING...", (15, 125), font, 0.7, (150, 150, 150), 2, cv2.LINE_AA)

   # 2. 선을 방금 찾아서 타겟 좌표를 향해 복귀 중일 때 (Status 8)
    elif state.status_code == 8:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 2)
        cv2.putText(frame, "TARGET LOCKED", (15, 125), font, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        
        if len(state.centroids) > 0:
            # ★ 시각화 과녁도 맨 위([-1])가 아닌 발밑 점([0])에 찍히도록 수정!
            tx, ty = int(state.centroids[0][0]), int(state.centroids[0][1])
            cv2.circle(frame, (tx, ty), 15, (0, 165, 255), 2)
            cv2.drawMarker(frame, (tx, ty), (0, 165, 255), cv2.MARKER_CROSS, 20, 2)
    # ==============================================================

    return frame

# ═══════════════════════════════════════════════════════
#  7. ROS2 노드
# ═══════════════════════════════════════════════════════

def main_ros2(ini_path: str = "settings.ini"):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    import json

    cfg = load_config(ini_path)

    class LineTrackerNode(Node):
        def __init__(self):
            super().__init__("line_tracker")
            self.cfg    = cfg
            self.bridge = CvBridge()
            self.frame_count = 0
            self.sub = self.create_subscription(
                Image, "/camera/image_raw", self.cb_image, 10
            )
            self.pub_state = self.create_publisher(String, "/line_tracker/state", 10)
            self.pub_debug = self.create_publisher(Image,  "/line_tracker/debug_image", 10)
            self.get_logger().info(f"LineTrackerNode started  cfg={ini_path}")

        def cb_image(self, msg: Image):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.cfg["flip_vertical"]:
                frame = cv2.flip(frame, 0)

            state, vis = analyze_frame(frame, self.cfg)

            payload = {
                "direction":     state.direction.value,
                "curvature":     state.curvature,
                "center_offset": state.center_offset,
                "confidence":    state.confidence,
                "n_markers":     len(state.centroids),
            }
            self.pub_state.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            debug_msg        = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)

            self.frame_count += 1
            if self.frame_count % self.cfg["print_every_n_frames"] == 0:
                self.get_logger().info(
                    f"[{self.frame_count}] {state.direction.value} "
                    f"a={state.curvature:.2e} conf={state.confidence:.2f}"
                )

    rclpy.init()
    node = LineTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# ═══════════════════════════════════════════════════════
#  8. 독립 실행 (웹캠 / 영상 파일 테스트)
# ═══════════════════════════════════════════════════════

def main_standalone(ini_path: str = "settings.ini"):
    cfg = load_config(ini_path)

    cap = cv2.VideoCapture(cfg["cam_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["cam_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["cam_height"])
    cap.set(cv2.CAP_PROP_FPS,          cfg["cam_fps"])

    writer = None
    if cfg["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg["save_video"], fourcc, cfg["cam_fps"],
                                 (cfg["cam_width"], cfg["cam_height"]))

    frame_count = 0
    prev_time   = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cfg["flip_vertical"]:
            frame = cv2.flip(frame, 0)

        state, vis = analyze_frame(frame, cfg)

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        cv2.putText(vis, f"FPS: {fps:.1f}", (vis.shape[1] - 120, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

        if cfg["show_window"]:
            cv2.imshow("Line Tracker (Poly2)", vis)
        if writer:
            writer.write(vis)

        frame_count += 1
        if frame_count % cfg["print_every_n_frames"] == 0:
            print(f"[{frame_count}] {state.direction.value} "
                  f"a={state.curvature:.2e} conf={state.confidence:.2f} "
                  f"offset={state.center_offset:+.0f}px  FPS={fps:.1f}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
            cfg["white_lower"][2] = min(255, cfg["white_lower"][2] + 5)
            print(f"[CFG] white_lower_v -> {cfg['white_lower'][2]}")
        elif key == ord("s"):
            cfg["white_lower"][2] = max(0, cfg["white_lower"][2] - 5)
            print(f"[CFG] white_lower_v -> {cfg['white_lower'][2]}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════
#  진입점
# ═══════════════════════════════════════════════════════

def main(args=None):
    """ROS2 entry point"""
    # 설정 파일 경로 지정
    my_ini_path = "/home/solhee/ros2_study/src/rnd/rnd/settings.ini"
    # 웹캠 화면을 바로 띄우는 방식으로 실행
    main_standalone(my_ini_path)

if __name__ == "__main__":
    import sys
    ini = next((a for a in sys.argv[1:] if a.endswith(".ini")), "settings.ini")
    if "--ros2" in sys.argv:
        main_ros2(ini)
    else:
        main_standalone(ini)