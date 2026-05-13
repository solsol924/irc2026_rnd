#!/usr/bin/env python3
"""
Hurdle Vision System - Intel RealSense D435i
=============================================
카메라 설치 : 바닥과 수직 (Top-Down 뷰), 높이 약 43cm
이동 방식   : 로봇이 전진함에 따라 허들이 화면(Y축) 위에서 아래로 이동
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque

# ============================================================
#  파라미터
# ============================================================

HURDLE_REAL_HEIGHT = 0.050   # 실제 허들 높이 (m)
CAMERA_HEIGHT_M    = 0.430   # 카메라 높이 (m)

# 상태 전환 기준 위치 (0.0: ROI 상단, 1.0: ROI 하단)
# 허들이 ROI의 75% 아래쪽으로 내려오면(로봇과 가까워지면) CROSSING으로 전환
CROSS_Y_RATIO = 0.75  

MIN_HURDLE_PIXELS = 1000     # 허들로 인식할 최소 픽셀 수 (노이즈 필터링)

ROI_X_CENTER = 0.50
ROI_WIDTH    = 0.65
ROI_Y_TOP    = 0.20
ROI_Y_BOTTOM = 0.98

DEPTH_MIN_M  = 0.10
DEPTH_MAX_M  = 3.00
HEIGHT_HISTORY_N  = 10
CALIB_FRAMES = 60            # 2초 (30fps 기준)

H_MIN           = 0.015      # 바닥보다 1.5cm 이상 솟아있으면 허들로 간주
H_MAX           = 0.120      # 최대 12cm까지만 허들로 간주

# 색상
DARK_BLUE         = (139, 0, 0)
COLOR_ROI_BOX     = (200, 50, 0)
COLOR_STATUS_OK   = (180, 180, 0)
COLOR_STATUS_WARN = (0, 140, 255)
COLOR_CROSSING    = (200, 0, 200)
COLOR_CALIB       = (0, 220, 220)


class State:
    CALIBRATING = "CALIBRATING"
    WALKING     = "WALKING"
    DETECTING   = "DETECTING"
    CROSSING    = "CROSSING"


class HurdleVisionSystem:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(cfg)
        self.align   = rs.align(rs.stream.color)

        ds = self.profile.get_device().first_depth_sensor()
        self.depth_scale = ds.get_depth_scale()

        self.spatial      = rs.spatial_filter()
        self.temporal     = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.7)
        self.spatial.set_option(rs.option.filter_smooth_delta, 8)

        self.state          = State.CALIBRATING
        self.height_history = deque(maxlen=HEIGHT_HISTORY_N)
        
        self.is_auto_mode   = True
        self.miss_count     = 0

        self.W   = 640
        self.H   = 480
        
        # 캘리브레이션 맵
        self.floor_depth_map = None   
        self.calib_buffer    = []
        self.calib_done      = False

        print("[HurdleVision] 탑뷰 수직 이동 모드 초기화 완료")
        print()
        print("  [r] 재캘리브레이션  [c] 수동 CROSSING  [w] 수동 WALKING  [a] 자동 모드  [q] 종료")

    def postprocess(self, depth_frame):
        f = self.spatial.process(depth_frame)
        f = self.temporal.process(f)
        f = self.hole_filling.process(f)
        return f.as_depth_frame()

    def _roi_bounds(self):
        x1 = int(self.W * (ROI_X_CENTER - ROI_WIDTH / 2))
        x2 = int(self.W * (ROI_X_CENTER + ROI_WIDTH / 2))
        y1 = int(self.H * ROI_Y_TOP)
        y2 = int(self.H * ROI_Y_BOTTOM)
        return x1, y1, x2, y2

    # ----------------------------------------------------------
    def calibrate_floor(self, depth_img):
        x1, y1, x2, y2 = self._roi_bounds()
        roi = depth_img[y1:y2, x1:x2].astype(float) * self.depth_scale
        
        # 비정상적인 0값이나 너무 먼 값은 nan 처리
        roi = np.where((roi > DEPTH_MIN_M) & (roi < DEPTH_MAX_M), roi, np.nan)
        self.calib_buffer.append(roi)

        progress = len(self.calib_buffer) / CALIB_FRAMES
        if len(self.calib_buffer) >= CALIB_FRAMES:
            stacked = np.stack(self.calib_buffer, axis=0)
            
            # 각 픽셀별 바닥의 기준 깊이를 저장 (완전한 2D 맵)
            self.floor_depth_map = np.nanmedian(stacked, axis=0)
            self.calib_done = True
            self.state = State.WALKING

            print(f"[캘리브레이션 완료] 바닥 형태 매핑 완료")

        return progress

    # ----------------------------------------------------------
    def analyze_hurdle(self, depth_img):
        """허들 감지 및 위치/높이 분석 (탑뷰 기반)"""
        if not self.calib_done:
            return False, None, None

        x1, y1, x2, y2 = self._roi_bounds()
        roi = depth_img[y1:y2, x1:x2].astype(float) * self.depth_scale
        
        # 바닥 기준 맵에서 현재 깊이를 뺌 (결과가 양수면 물체가 바닥보다 튀어나와 있다는 뜻)
        delta_depth = self.floor_depth_map - roi
        
        # 지정된 높이 구간(1.5cm ~ 12cm)에 해당하는 픽셀만 필터링
        hurdle_mask = (delta_depth > H_MIN) & (delta_depth < H_MAX)
        
        ys, xs = np.where(hurdle_mask)
        
        if len(ys) > MIN_HURDLE_PIXELS:
            self.miss_count = 0
            # 허들 픽셀들의 중앙값을 높이로 채택
            measured_h = np.median(delta_depth[hurdle_mask])
            self.height_history.append(float(measured_h))
            hurdle_h = float(np.median(self.height_history))
            
            # 화면상 허들의 Y 중심점 계산 (0.0 ~ 1.0)
            roi_height = y2 - y1
            hurdle_y_pixel = np.median(ys) 
            y_ratio = hurdle_y_pixel / roi_height
            
            return True, hurdle_h, y_ratio
        else:
            self.miss_count += 1
            if self.miss_count > 10:  # 0.3초 이상 안 보이면 리셋
                self.height_history.clear()
            return False, None, None

    # ----------------------------------------------------------
    def update_state(self, is_detected, y_ratio):
        if self.state == State.CALIBRATING:
            return
        
        prev = self.state

        # 1. 화면에 허들이 없으면 WALKING
        if not is_detected:
            self.state = State.WALKING
        # 2. 화면에 허들이 들어왔을 때
        else:
            # 허들의 Y 위치가 설정한 선(CROSS_Y_RATIO)을 넘어섰다면 CROSSING
            if y_ratio >= CROSS_Y_RATIO:
                self.state = State.CROSSING
            # 그 전까지는 다 DETECTING
            else:
                self.state = State.DETECTING

        if prev != self.state:
            print(f"[FSM] {prev} → {self.state}")

    # ----------------------------------------------------------
    def make_depth_minimap(self, depth_img, roi_rect):
        x1, y1, x2, y2 = roi_rect
        roi_d = depth_img[y1:y2, x1:x2].astype(float) * self.depth_scale
        
        # 시각화를 위한 정규화
        clipped = np.clip(roi_d, DEPTH_MIN_M, 1.0)
        norm = ((1.0 - clipped) / (1.0 - DEPTH_MIN_M) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

        # 허들 부분 덧칠하기 (하늘색)
        if self.calib_done:
            delta = self.floor_depth_map - roi_d
            hurdle_px = (delta > H_MIN) & (delta < H_MAX)
            colored[hurdle_px] = (255, 255, 0)  # BGR 하늘색

        return cv2.resize(colored, (210, 158))

    # ----------------------------------------------------------
    def draw_overlay(self, color_img, depth_img, is_detected, hurdle_h, y_ratio,
                     roi_rect, calib_progress=0.0):
        img = color_img.copy()
        x1, y1, x2, y2 = roi_rect

        # ROI 박스
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_ROI_BOX, 2)
        
        # CROSSING 기준선 그리기 (점선 느낌)
        cross_y = int(y1 + (y2 - y1) * CROSS_Y_RATIO)
        cv2.line(img, (x1, cross_y), (x2, cross_y), (100, 100, 100), 2)
        cv2.putText(img, "CROSSING LINE", (x1 + 5, cross_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        # 캘리브레이션 바
        if self.state == State.CALIBRATING:
            bx1, bx2 = 20, self.W - 20
            by = self.H // 2
            bar_w = int((bx2 - bx1) * calib_progress)
            cv2.rectangle(img, (bx1, by - 28), (bx2, by + 28), (30, 30, 30), -1)
            cv2.rectangle(img, (bx1, by - 28), (bx1 + bar_w, by + 28), COLOR_CALIB, -1)
            cv2.putText(img, f"Floor Calibrating {calib_progress*100:.0f}%",
                        (bx1 + 10, by + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 0), 2, cv2.LINE_AA)

        # 상단 텍스트 (위치 및 높이 정보)
        if is_detected and y_ratio is not None:
            pos_txt = f"Hurdle Pos: {y_ratio*100:.1f} %"
            h_txt   = f"Hurdle H: {hurdle_h*100:.1f} cm"
            h_color = (0, 150, 0)
        else:
            pos_txt = "Hurdle Pos: -- %"
            h_txt   = "Hurdle H: Measuring..."
            h_color = DARK_BLUE

        cv2.putText(img, pos_txt, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, DARK_BLUE, 2, cv2.LINE_AA)
        cv2.putText(img, h_txt, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, h_color, 2, cv2.LINE_AA)

        # 하단 상태 표시 (AUTO 텍스트 제거)
        sc = {State.WALKING:     COLOR_STATUS_OK,
              State.DETECTING:   COLOR_STATUS_WARN,
              State.CROSSING:    COLOR_CROSSING,
              State.CALIBRATING: COLOR_CALIB}.get(self.state, (200, 200, 200))
        
        cv2.putText(img, f"State: {self.state}", (12, self.H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, sc, 2, cv2.LINE_AA)

        if self.state == State.CROSSING:
            cv2.rectangle(img, (4, 4), (self.W - 4, self.H - 4), COLOR_CROSSING, 5)

        # 깊이 미니맵
        mini = self.make_depth_minimap(depth_img, roi_rect)
        mh, mw = mini.shape[:2]
        img[8:8 + mh, self.W - mw - 8:self.W - 8] = mini

        return img

    # ----------------------------------------------------------
    def run(self):
        cv2.namedWindow("Hurdle Vision", cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                frames  = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned = self.align.process(frames)
                df = aligned.get_depth_frame()
                cf = aligned.get_color_frame()
                if not df or not cf:
                    continue

                df        = self.postprocess(df)
                depth_img = np.asanyarray(df.get_data())
                color_img = np.asanyarray(cf.get_data())

                calib_progress = 0.0
                roi_rect = self._roi_bounds()

                if self.state == State.CALIBRATING:
                    calib_progress = self.calibrate_floor(depth_img)
                    vis = self.draw_overlay(color_img, depth_img, False, None, None,
                                           roi_rect, calib_progress)
                    cv2.imshow("Hurdle Vision", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break
                    continue

                # 탑뷰 허들 분석
                is_detected, hurdle_h, y_ratio = self.analyze_hurdle(depth_img)

                # 자동 모드일 때만 FSM 로직 작동
                if self.is_auto_mode:
                    self.update_state(is_detected, y_ratio)

                vis = self.draw_overlay(color_img, depth_img, is_detected, hurdle_h, y_ratio, roi_rect)
                cv2.imshow("Hurdle Vision", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.state = State.CROSSING
                    self.is_auto_mode = False
                    print("[FSM] 수동 개입 → CROSSING")
                elif key == ord('w'):
                    self.state = State.WALKING
                    self.is_auto_mode = False
                    print("[FSM] 수동 개입 → WALKING")
                elif key == ord('a'):
                    self.is_auto_mode = True
                    print("[FSM] 자동 모드 활성화")
                elif key == ord('r'):
                    self.state        = State.CALIBRATING
                    self.is_auto_mode = True
                    self.calib_buffer = []
                    self.calib_done   = False
                    self.height_history.clear()

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    system = HurdleVisionSystem()
    system.run()

if __name__ == "__main__":
    main()