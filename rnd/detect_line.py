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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════
#  열거형 / 데이터 클래스
# ═══════════════════════════════════════════════════════

class Direction(Enum):
    STRAIGHT = "직선"
    LEFT     = "좌회전"
    RIGHT    = "우회전"
    UNKNOWN  = "판별불가"


@dataclass
class TrackState:
    direction:      Direction
    curvature:      float                   # 2차항 계수 a
    center_offset:  float                   # 근거리 라인 중심 ~ 로봇 중심 (px)
    poly_coeffs:    Optional[np.ndarray]    # [a, b, c]
    confidence:     float
    centroids:      list = field(default_factory=list)
    band_centroids: list = field(default_factory=list)


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
    rect  = cv2.minAreaRect(contour)
    w, h_ = rect[1]
    if w < 1 or h_ < 1:
        return False
    aspect = max(w, h_) / min(w, h_)
    return cfg["min_aspect_ratio"] <= aspect <= cfg["max_aspect_ratio"]


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
#  3. 2차 다항식 피팅  x = a*y^2 + b*y + c
# ═══════════════════════════════════════════════════════

def fit_poly2(points):
    if len(points) < 3:
        return None
    pts = np.array(points, dtype=np.float64)
    ys, xs = pts[:, 1], pts[:, 0]
    y_mean = ys.mean()
    y_std  = ys.std() if ys.std() > 1e-6 else 1.0
    yn     = (ys - y_mean) / y_std
    try:
        a_n, b_n, c_n = np.polyfit(yn, xs, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    s = y_std
    a = a_n / s**2
    b = -2 * a_n * y_mean / s**2 + b_n / s
    c = a_n * y_mean**2 / s**2 - b_n * y_mean / s + c_n
    return np.array([a, b, c])


def poly_confidence(points, coeffs):
    if len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float64)
    ys, xs = pts[:, 1], pts[:, 0]
    a, b, c = coeffs
    mean_res = np.abs(xs - (a * ys**2 + b * ys + c)).mean()
    return float(max(0.0, 1.0 - mean_res / 50.0))


# ═══════════════════════════════════════════════════════
#  4. 방향 판별
# ═══════════════════════════════════════════════════════

def determine_direction(centroids, band_pts, coeffs, frame_w, cfg) -> TrackState:
    robot_cx = frame_w / 2.0
    if not centroids:
        return TrackState(Direction.UNKNOWN, 0.0, 0.0, None, 0.0)

    center_offset = centroids[0][0] - robot_cx

    if coeffs is None:
        if abs(center_offset) < frame_w * 0.05:
            dir_ = Direction.STRAIGHT
        elif center_offset < 0:
            dir_ = Direction.LEFT
        else:
            dir_ = Direction.RIGHT
        return TrackState(dir_, 0.0, center_offset, None, 0.15, centroids, band_pts)

    a, b, c    = coeffs
    confidence = poly_confidence(band_pts if band_pts else centroids, coeffs)
    str_thr    = cfg["straight_curvature_thresh"]

    if confidence < cfg["min_confidence"]:
        direction = Direction.UNKNOWN
    elif abs(a) < str_thr:
        direction = Direction.STRAIGHT
    elif a > 0:
        direction = Direction.RIGHT
    else:
        direction = Direction.LEFT

    return TrackState(direction, float(a), center_offset, coeffs,
                      confidence, centroids, band_pts)


# ═══════════════════════════════════════════════════════
#  5. 통합 파이프라인
# ═══════════════════════════════════════════════════════

def analyze_frame(frame: np.ndarray, cfg: dict) -> tuple[TrackState, np.ndarray]:
    h, w = frame.shape[:2]
    roi_top    = int(h * cfg["roi_top_ratio"])
    roi_bottom = int(h * cfg["roi_bottom_ratio"])
    roi_left   = int(w * cfg["roi_left_ratio"])
    roi_right  = int(w * cfg["roi_right_ratio"])

    roi       = frame[roi_top:roi_bottom, roi_left:roi_right]
    mask_roi  = preprocess(roi, cfg)
    centroids = extract_centroids(mask_roi, cfg, offset_x=roi_left, offset_y=roi_top)
    band_pts  = band_sample(centroids, roi_top, roi_bottom, cfg["n_bands"])

    fit_src = band_pts if len(band_pts) >= cfg["min_points_for_poly"] else centroids
    coeffs  = fit_poly2(fit_src) if len(fit_src) >= cfg["min_points_for_poly"] else None

    state = determine_direction(centroids, band_pts, coeffs, w, cfg)

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
    Direction.UNKNOWN:  (120, 120, 120),
}
ARROW = {
    Direction.STRAIGHT: "UP   STRAIGHT",
    Direction.LEFT:     "LEFT  LEFT-TURN",
    Direction.RIGHT:    "RIGHT RIGHT-TURN",
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

    cv2.rectangle(frame, (8, 8), (370, 148), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (370, 148), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, ARROW[state.direction],
                (18, 52), font, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Curvature a: {state.curvature:.2e}",
                (18, 82), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Conf: {state.confidence:.2f}  Offset: {state.center_offset:+.0f}px",
                (18, 108), font, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Markers: {len(state.centroids)}  Bands: {len(state.band_centroids)}",
                (18, 132), font, 0.50, (120, 120, 120), 1, cv2.LINE_AA)
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

if __name__ == "__main__":
    import sys
    ini = next((a for a in sys.argv[1:] if a.endswith(".ini")), "settings.ini")
    if "--ros2" in sys.argv:
        main_ros2(ini)
    else:
        main_standalone(ini)