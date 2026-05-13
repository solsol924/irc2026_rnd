#!/usr/bin/env python3
import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import Counter
from robot_msgs.msg import BallResult, MotionEnd # type: ignore
from message_filters import Subscriber, ApproximateTimeSynchronizer # type: ignore  동기화용
from rcl_interfaces.msg import SetParametersResult

CAM1, CAM2 = 1, 2
BALL, HOOP = 1, 2

camera_width = 640
camera_height = 480

class LineListenerNode(Node): #################################################################### 판단 프레임 수 바꿀 때 yolo_cpp 도 고려해라~~~
    def __init__(self):
        super().__init__('line_subscriber')

        self.roi_x_start = int(camera_width * 0 // 5)
        self.roi_x_end = int(camera_width * 5 // 5)
        self.roi_y_start = int(camera_height * 6 // 12)
        self.roi_y_end = int(camera_height * 12 // 12)

        self.roi_h = self.roi_y_end - self.roi_y_start
        self.roi_w = self.roi_x_end - self.roi_x_start

        self.zandi_x = 320
        self.zandi_y = 240

        self.pick_x = 315
        self.pick_y = 287

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)
        
        self.last_avg_text = "AVG: --- ms | FPS: --"
        self.last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력
        self.backboard_score_text = 'Miss' # H

        # 추적
        self.last_cx_ball = self.last_cy_ball = self.last_z_ball = self.last_radius = self.last_radius = None
        self.last_avg_cy_ball = 0 # 화면 전환용
        self.ball_lost = 0 # B

        self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
        self.last_cx_hoop = self.last_cy_hoop = self.last_z_hoop = self.last_yaw = None
        self.last_box_hoop = None
        self.last_avg_cx_hoop = 0 
        self.hoop_lost = 0 # H

        self.last_res = 99

        # 변수
        self.draw_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.fx, self.fy = 607.0, 606.0   # ros2 topic echo /camera/color/camera_info - 카메라 고유값
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.last_band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype = np.uint8)  # 영행렬
        
        self.depth_max_ball = 1500.0  # mm 기준 마스크 거리 1.5m
        # self.depth_max_hoop = 1500.0  # 1.5m  << 여기 전에 감지한 값이랑 비교해서 조절되게끔
        self.depth_min = 50.0  # 5cm
        self.depth_scale = 0.001    # mm >> m 변환용 
        
        self.collecting = False     # 수집 중 여부
        self.armed = False          # motion_end 방어~!

        self.window_id = 0
        self.frame_idx = 0       
        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        self.collecting_frames = 15
        
        self.cam1_ball_count = 0
        self.cam2_miss_count = 0
        self.hoop_miss_count = 0
        self.pick_attempt = 0

        self.picked = False
        self.ball_saw_once = False
        self.ball_never_seen = True
        self.hoop_near_by = False

        self.last_avg_cx_ball = 0
        self.last_avg_cy_ball = 0

        self.ball_valid_list = []
        self.ball_cx_list = []
        self.ball_cy_list = []
        self.ball_dis_list = [] # B

        self.hoop_valid_list = [] # H
        self.hoop_cx_list = []
        self.hoop_dis_list = []
        self.yaw_list = []

        self.hoop_r = 100 # 골대 반지름
        self.throwing_range_z = 400 # 공 던지는 거리
        self.goal_range = 50 # 공 던질 때 떨어지는 곳 오차범위

        self.bridge = CvBridge()   
        
        # CAM1
        self.cam1_color_sub = Subscriber(self, Image, '/cam1/color/image_raw')
        self.cam1_depth_sub = Subscriber(self, Image, '/cam1/aligned_depth_to_color/image_raw') 
        self.cam1_sync = ApproximateTimeSynchronizer([self.cam1_color_sub, self.cam1_depth_sub], queue_size=5, slop=0.1) # 컬러, 깊이 동기화
        self.cam1_sync.registerCallback(self.cam1_image_callback)

        # CAM2
        self.cam2_color_sub = self.create_subscription(  
            Image,
            '/cam2/color/image_raw',  #  640x480 / 15fps
            self.cam2_image_callback, 10)
        
        self.motion_end_sub = self.create_subscription(  # 모션엔드
            MotionEnd,                   
            '/motion_end',                   
            self.motion_callback, 10) 
        
        # 퍼블리셔
        self.ball_result_pub = self.create_publisher(BallResult, '/ball_result', 10)

        # 파라미터 선언 B
        self.declare_parameter("cam_mode", CAM1)
        self.declare_parameter("cam1_mode", BALL)

        self.declare_parameter("orange_h_low_1", 8) # 주황 8, 20 < 동방 기준임
        self.declare_parameter("orange_h_high_1", 15)  
        self.declare_parameter("orange_s_low_1", 150) # 채도  40, 255
        self.declare_parameter("orange_s_high_1", 255)  
        self.declare_parameter("orange_v_low_1", 170) # 밝기 50, 255
        self.declare_parameter("orange_v_high_1", 255)

        self.declare_parameter("orange_h_low_2", 1) # 주황 8, 20 < 동방 기준임
        self.declare_parameter("orange_h_high_2", 30)  
        self.declare_parameter("orange_s_low_2", 100) # 채도  40, 255
        self.declare_parameter("orange_s_high_2", 255)  
        self.declare_parameter("orange_v_low_2", 150) # 밝기 50, 255
        self.declare_parameter("orange_v_high_2", 255)

        # 파라미터 선언 H
        self.declare_parameter('red_h1_low', 0) # 빨강 구간1
        self.declare_parameter('red_h1_high', 10)
        self.declare_parameter('red_h2_low', 160)   # 빨강 구간2
        self.declare_parameter('red_h2_high', 180)
        self.declare_parameter('red_s_low', 80)  # 채도
        self.declare_parameter('red_v_low', 60)  # 밝기
        
        self.declare_parameter('white_s_high', 80) # 80
        self.declare_parameter('white_v_low', 40) # 40

        self.declare_parameter('band_top_ratio', 0.15)   # 백보드 h x 0.15
        self.declare_parameter('band_side_ratio', 0.10)   # w x 0.10

        self.declare_parameter('red_ratio_min', 0.55)     # 백보드 영역
        self.declare_parameter('white_inner_ratio_min', 0.50)  
        self.declare_parameter('backboard_area', 500)

        self.declare_parameter('depth_max_hoop', 1500)

        # self.declare_parameter('prox_px', 30)
        self.declare_parameter('expand_px', 13)
        # self.declare_parameter('shrink_px', 13)

        # 파라미터 적용 B
        self.cam_mode = self.get_parameter("cam_mode").value
        self.cam1_mode = self.get_parameter("cam1_mode").value

        self.orange_h_low_1 = self.get_parameter("orange_h_low_1").value
        self.orange_h_high_1 = self.get_parameter("orange_h_high_1").value
        self.orange_s_low_1 = self.get_parameter("orange_s_low_1").value
        self.orange_s_high_1 = self.get_parameter("orange_s_high_1").value
        self.orange_v_low_1 = self.get_parameter("orange_v_low_1").value
        self.orange_v_high_1 = self.get_parameter("orange_v_high_1").value

        self.orange_h_low_2 = self.get_parameter("orange_h_low_2").value
        self.orange_h_high_2 = self.get_parameter("orange_h_high_2").value
        self.orange_s_low_2 = self.get_parameter("orange_s_low_2").value
        self.orange_s_high_2 = self.get_parameter("orange_s_high_2").value
        self.orange_v_low_2 = self.get_parameter("orange_v_low_2").value
        self.orange_v_high_2 = self.get_parameter("orange_v_high_2").value

        # 파라미터 적용 H
        self.red_h1_low = self.get_parameter('red_h1_low').value
        self.red_h1_high = self.get_parameter('red_h1_high').value
        self.red_h2_low = self.get_parameter('red_h2_low').value
        self.red_h2_high = self.get_parameter('red_h2_high').value
        self.red_s_low = self.get_parameter('red_s_low').value
        self.red_v_low = self.get_parameter('red_v_low').value

        self.white_s_high = self.get_parameter('white_s_high').value
        self.white_v_low = self.get_parameter('white_v_low').value

        self.band_top_ratio = self.get_parameter('band_top_ratio').value
        self.band_side_ratio = self.get_parameter('band_side_ratio').value

        self.red_ratio_min = self.get_parameter('red_ratio_min').value
        self.white_inner_ratio_min = self.get_parameter('white_inner_ratio_min').value
        self.backboard_area = self.get_parameter('backboard_area').value

        self.depth_max_hoop = self.get_parameter('depth_max_hoop').value 

        # self.prox_px = self.get_parameter('prox_px').value
        self.expand_px = self.get_parameter('expand_px').value
        # self.shrink_px = self.get_parameter('shrink_px').value

        self.add_on_set_parameters_callback(self.param_callback)

        self.lower_hsv_ball = np.array([self.orange_h_low_1, self.orange_s_low_1, self.orange_v_low_1], dtype=np.uint8) # 색공간 미리 선언
        self.upper_hsv_ball = np.array([self.orange_h_high_1, self.orange_s_high_1, self.orange_v_high_1], dtype=np.uint8)
        self.lower_hsv_ball2 = np.array([self.orange_h_low_2, self.orange_s_low_2, self.orange_v_low_2], dtype=np.uint8)  # 두 번째 필터용
        self.upper_hsv_ball2 = np.array([self.orange_h_high_2, self.orange_s_high_2, self.orange_v_high_2], dtype=np.uint8)  # 두 번째 필터용
        self.hsv = None 

        self.apply_mode_layout()

        # 클릭
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ball Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Detection', self.on_click)
        cv2.setMouseCallback('Ball Detection', self.on_click)

    def apply_mode_layout(self):
        if self.cam_mode == CAM1:
            if self.cam1_mode == BALL:
                self.roi_x_start, self.roi_x_end = 0, camera_width
                self.roi_y_start, self.roi_y_end = camera_height*1//12, camera_height
            else:  # HOOP
                self.roi_x_start, self.roi_x_end = 0, camera_width
                self.roi_y_start, self.roi_y_end = camera_height*4//12, camera_height*10//12 # 크기 바꾸면 안 됨
        else:  # CAM2
            self.roi_x_start, self.roi_x_end = 0, camera_width
            self.roi_y_start, self.roi_y_end = 0, camera_height

        # ROI 마스크 재생성
        h = self.roi_y_end - self.roi_y_start
        w = self.roi_x_end - self.roi_x_start
        self.last_band_mask = np.zeros((h, w), dtype=np.uint8)

    def param_callback(self, params):
        old_cam_mode = self.cam_mode
        old_cam1_mode = self.cam1_mode
        for p in params:
            if p.name == "cam_mode":   self.cam_mode = int(p.value)
            elif p.name == "cam1_mode": self.cam1_mode = int(p.value)
            elif p.name == "orange_h_low_1":   self.orange_h_low_1 = int(p.value)
            elif p.name == "orange_h_high_1": self.orange_h_high_1 = int(p.value)
            elif p.name == "orange_s_low_1":  self.orange_s_low_1 = int(p.value)
            elif p.name == "orange_s_high_1": self.orange_s_high_1 = int(p.value)
            elif p.name == "orange_v_low_1":  self.orange_v_low_1 = int(p.value)
            elif p.name == "orange_v_high_1": self.orange_v_high_1 = int(p.value)
            elif p.name == "orange_h_low_2":   self.orange_h_low_2 = int(p.value)
            elif p.name == "orange_h_high_2": self.orange_h_high_2 = int(p.value)
            elif p.name == "orange_s_low_2":  self.orange_s_low_2 = int(p.value)
            elif p.name == "orange_s_high_2": self.orange_s_high_2 = int(p.value)
            elif p.name == "orange_v_low_2":  self.orange_v_low_2 = int(p.value)
            elif p.name == "orange_v_high_2": self.orange_v_high_2 = int(p.value)
            elif p.name == "red_h1_low":    self.red_h1_low = int(p.value)
            elif p.name == "red_h1_high":   self.red_h1_high = int(p.value)
            elif p.name == "red_h2_low":    self.red_h2_low = int(p.value)
            elif p.name == "red_h2_high":   self.red_h2_high = int(p.value)
            elif p.name == "red_s_low":     self.red_s_low = int(p.value)
            elif p.name == "red_v_low":     self.red_v_low = int(p.value)  
            elif p.name == "white_s_high":  self.white_s_high = int(p.value)
            elif p.name == "white_v_low":   self.white_v_low = int(p.value)
            elif p.name == "band_top_ratio":  self.band_top_ratio = float(p.value)
            elif p.name == "band_side_ratio": self.band_side_ratio = float(p.value)
            elif p.name == "red_ratio_min":   self.red_ratio_min = float(p.value)
            elif p.name == "white_inner_ratio_min": self.white_inner_ratio_min = float(p.value)
            elif p.name == "backboard_area":  self.backboard_area = int(p.value)
            elif p.name == "depth_max_hoop":  self.depth_max_hoop = int(p.value)
            # elif p.name == "prox_px":  self.prox_px = int(p.value)
            elif p.name == "expand_px":  self.expand_px = int(p.value)
            # elif p.name == "shrink_px":  self.shrink_px = int(p.value)

        # HSV 경계값 배열 갱신
        self.lower_hsv_ball = np.array([self.orange_h_low_1, self.orange_s_low_1, self.orange_v_low_1], dtype=np.uint8)
        self.upper_hsv_ball = np.array([self.orange_h_high_1, self.orange_s_high_1, self.orange_v_high_1], dtype=np.uint8)
        self.lower_hsv_ball2 = np.array([self.orange_h_low_2, self.orange_s_low_2, self.orange_v_low_2], dtype=np.uint8)  # 두 번째 필터용
        self.upper_hsv_ball2 = np.array([self.orange_h_high_2, self.orange_s_high_2, self.orange_v_high_2], dtype=np.uint8)  # 두 번째 필터용

        # 모드/서브모드가 바뀌었으면 ROI 등 즉시 적용
        if (old_cam_mode != self.cam_mode) or (old_cam1_mode != self.cam1_mode):
            self.apply_mode_layout()
            # 진행 중 윈도우 정리
            self.collecting = False
            self.frames_left = 0

        return SetParametersResult(successful=True)
        
    def on_click(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        if not (self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end):
            return

        rx = x - self.roi_x_start
        ry = y - self.roi_y_start
        k = 1  # (3,3)

        y0, y1 = max(0, ry - k), min(self.roi_y_end - self.roi_y_start, ry + k + 1)
        x0, x1 = max(0, rx - k), min(self.roi_x_end - self.roi_x_start, rx + k + 1)
        patch = self.hsv[y0:y1, x0:x1].reshape(-1,3)

        H, S, V = np.mean(patch, axis=0).astype(int)
        self.get_logger().info(f"[Pos] x={x - self.zandi_x}, y={-(y - self.zandi_y)} | HSV=({H},{S},{V})")

    def motion_callback(self, msg: MotionEnd): # 모션 끝 같이 받아오기 (중복 방지)
        if bool(msg.motion_end_detect):
            self.armed = True
            self.get_logger().info("Subscribed /motion_end !!!!!!!!!!!!!!!!!!!!!!!!")

    def cam1_image_callback(self, cam1_color_msg: Image, cam1_depth_msg: Image): # yolo_cpp에서 토픽 보내면 실핼

        if self.cam_mode != CAM1:
            return
        
        start_time = time.time()
        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(cam1_color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(cam1_depth_msg, desired_encoding='passthrough').astype(np.float32)

        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        roi_depth = depth[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        if self.cam1_mode == BALL: ##################################################################################################3
            if not self.collecting:
                if self.armed:
                    self.collecting = True
                    self.armed = False
                    self.frames_left = self.collecting_frames
                    self.window_id += 1
        
                    # 초기화
                    self.ball_valid_list.clear()
                    self.ball_cx_list.clear()
                    self.ball_cy_list.clear()
                    self.ball_dis_list.clear()
                    self.frame_idx = 0

                    self.last_cx_ball = self.last_cy_ball = self.last_radius = self.last_z_ball = None
                    self.ball_lost = 0
                    self.cam2_miss_count = 0

                    self.line_start_time = time.time()

                    self.draw_color = (0, 255, 0)
                    self.rect_color = (0, 255, 0)

                    self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames in CAM{self.cam_mode}')

            if self.collecting:
                self.frame_idx += 1
                self.get_logger().info(f"step {self.frame_idx}")
            
                self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
                raw_mask = cv2.inRange(self.hsv, self.lower_hsv_ball, self.upper_hsv_ball)
                raw_mask[roi_depth >= self.depth_max_ball] = 0
                raw_mask[roi_depth <= self.depth_min] = 0

                mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                contours_large = [cnt for cnt in contours if cv2.contourArea(cnt) >= 200]

                if len(contours_large) != 0:
                    print("11")

                    expanded_mask = np.zeros_like(mask)
                    cv2.drawContours(expanded_mask, contours_large, -1, 255, -1)  # 큰 컨투어 전부 채우기

                    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))  # 20픽셀 팽창
                    expanded_mask = cv2.dilate(expanded_mask, expand_kernel)

                    second_mask = cv2.inRange(self.hsv, self.lower_hsv_ball2, self.upper_hsv_ball2)
                    second_mask = cv2.bitwise_and(second_mask, expanded_mask)  # 확장된 영역 내에서만 필터 적용

                    second_mask = cv2.morphologyEx(second_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                    second_mask = cv2.morphologyEx(second_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

                    contours_refined, _ = cv2.findContours(second_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    best_cnt_ball = None
                    best_ratio_ball = 0.5
                    best_cx_ball = best_cy_ball = best_z_ball = best_radius = None

                    for cnt in contours_refined:
                        area = cv2.contourArea(cnt)
                        print(f"area: {area}")
                        if area > 300:
                            print("22")
                            (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                            circle_area = circle_r * circle_r * math.pi
                            ratio = area / circle_area
                            if ratio > best_ratio_ball:
                                best_cnt_ball = cnt
                                best_ratio_ball = ratio
                                best_cx_ball = int(x + self.roi_x_start)
                                best_cy_ball = int(y + self.roi_y_start)
                                best_radius = int(circle_r)

                else:
                    best_cnt_ball = None               

                # 검출 결과
                if best_cnt_ball is not None: # 원 탐지를 했다면
                    self.ball_lost = 0
                    
                    x1 = max(best_cx_ball - self.roi_x_start - 1, 0)
                    x2 = min(best_cx_ball - self.roi_x_start + 2, self.roi_x_end - self.roi_x_start)
                    y1 = max(best_cy_ball - self.roi_y_start - 1, 0)
                    y2 = min(best_cy_ball - self.roi_y_start + 2, self.roi_y_end - self.roi_y_start)

                    roi_patch = roi_depth[y1:y2, x1:x2]
                    ball_valid = np.isfinite(roi_patch) & (roi_patch > self.depth_min) & (roi_patch < self.depth_max_ball)
                    if np.count_nonzero(ball_valid) >= 1:
                        best_z_ball = float(np.mean(roi_patch[ball_valid]))
                        is_ball_valid = True
                        self.get_logger().info(f"[Ball] Found! | {best_cx_ball}, {best_cy_ball}, dist = {best_z_ball}")

                        self.last_z_ball = best_z_ball
                    else:
                        best_z_ball = -1 
                        is_ball_valid = False
                        self.get_logger().info(f"[Ball] Invalid depth! | {best_cx_ball}, {best_cy_ball}")
                    
                    # 이전 위치 업데이트
                    self.last_cx_ball = best_cx_ball
                    self.last_cy_ball = best_cy_ball
                    self.last_radius = best_radius
                    
                    self.rect_color = (0, 255, 0)
                    self.draw_color = (255, 0, 0)

                    cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)

                else:
                    if self.ball_lost < 3 and self.last_cx_ball is not None:
                        self.ball_lost += 1

                        best_cx_ball = self.last_cx_ball
                        best_cy_ball = self.last_cy_ball
                        best_z_ball = self.last_z_ball  
                        best_radius = self.last_radius

                        self.draw_color = (0, 255, 255)
                        self.rect_color = (255, 0, 0)

                        is_ball_valid = True
                        self.get_logger().info(f"[Ball] Lost! | {best_cx_ball}, {best_cy_ball}, dist = {best_z_ball}")

                        cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)
                    else:
                        self.ball_lost = 3

                        best_cx_ball = best_cy_ball = best_z_ball = best_radius = None
                        self.last_cx_ball = self.last_cy_ball = self.last_radius = self.last_z_ball = None
                        
                        self.rect_color = (0, 0, 255)

                        is_ball_valid = False
                        self.get_logger().info(f"[Ball] Miss!")

                self.frames_left -= 1
                self.ball_valid_list.append(is_ball_valid)
                self.ball_cx_list.append(best_cx_ball)
                self.ball_cy_list.append(best_cy_ball)
                self.ball_dis_list.append(best_z_ball)

                if self.frames_left <= 0:
                    result = self.ball_valid_list.count(True) >= 5
                    process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                    if result == True: # 공 찾았을 때
                        cxs = [a for s, a in zip(self.ball_valid_list, self.ball_cx_list) if s == result and a is not None]
                        cys = [a for s, a in zip(self.ball_valid_list, self.ball_cy_list) if s == result and a is not None]
                        dists = [a for s, a in zip(self.ball_valid_list, self.ball_dis_list) if s == result and a is not None]
                        avg_cx = int(round(np.mean(cxs)))
                        avg_cy = int(round(np.mean(cys)))
                        avg_dis = np.mean(dists) * self.depth_scale

                        # angle = int(round(-math.degrees(math.atan2(avg_cx - self.zandi_x, -avg_cy + 380))))  # 수정 완
                        angle = int(round(math.degrees(math.atan((avg_cx - self.cx_intr) / self.fx))))
                        self.last_avg_cy_ball = avg_cy # 다음 프레임에 판단용
                        
                        if avg_cy <= 320:
                            if angle >= 9.6: # 우회전
                                res = 30 
                            elif angle <= -9.6: # 좌회전
                                res = 29
                            else: # 직진
                                res = 1
                        else:
                            if angle >= 9.6: # 우회전
                                res = 3 
                            elif angle <= -9.6: # 좌회전
                                res = 2
                            else: # 직진
                                res = 1

                        self.get_logger().info(f"[Ball] Done: CAM1 found ball | {avg_cx}, {avg_cy}, dis: {avg_dis:.2f}, "
                                            f"angle= {angle}")
                        self.last_position_text = f"[Ball] Position: {avg_cx}, {avg_cy}"
                        self.cam1_ball_count += 1

                    else: # 공 못 찾았는데
                        if self.last_avg_cy_ball >= 320 and self.cam1_ball_count >= 1: # 그 전까지 공을 보고 있었고 공이 대충 밑에 있었다면
                            if self.last_avg_cy_ball >= 420:
                                res = 12
                            else:
                                res = 27
                            angle = 0
                            self.cam_mode = CAM2 # 2번 캠으로 ㄱㄱ
                            self.last_avg_cy_ball = 0

                            self.get_logger().info(f"[Ball] CAM1 Missed, CAM2 will find,,,")
                            
                            self.last_position_text = "[Ball] Changing to Cam2,,"
                            self.cam1_ball_count = 0

                            self.last_cx_ball = self.last_cy_ball = self.last_z_ball = self.last_radius = None
                            self.backboard_score_text = "Ball Now"

                            self.apply_mode_layout()

                        elif self.cam1_ball_count == 1:  # 한 번 찾았었는데 놓쳤다면
                            self.get_logger().info(f"[Ball] finding one more")
                            res = 97 # 뒤로 가서 다시 봐봐
                            angle = 0

                            self.last_position_text = "[Ball] finding one more"
                            self.cam1_ball_count = 0
                        
                        else: # 두 번 연속 못 찾았거나 공이 없거나
                            self.get_logger().info(f"[Ball] No ball detected")
                            res = 99 
                            angle = 0
                            
                            self.last_position_text = "[Ball] No ball"
                            self.cam1_ball_count = 0

                    # 퍼블리시
                    msg_out = BallResult()
                    msg_out.res = res
                    msg_out.angle = abs(angle)
                    self.ball_result_pub.publish(msg_out)

                    self.get_logger().info(f"res= {res}")
                    self.get_logger().info(f"frames= {len(self.ball_valid_list)}, wall= {process_time*1000:.1f} ms")
                    
                    # 리셋
                    self.collecting = False
                    self.frames_left = 0
                    self.frame_idx = 0
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 여기까지 self.frames_left <= 10 조건만 돌아가게. 왜와이: 첫 5프레임은 버리고
            # else:
            #     self.last_position_text = "In move"

            cv2.line(frame, (self.roi_x_start, 320), (self.roi_x_end, 320), (250, 122, 122), 2)

        elif self.cam1_mode == HOOP:
            if not self.collecting:
                if self.armed:
                    self.collecting = True
                    self.armed = False
                    self.frames_left = self.collecting_frames
                    self.window_id += 1
            
                    # 초기화
                    self.hoop_valid_list.clear()
                    self.hoop_cx_list.clear()
                    self.hoop_dis_list.clear()
                    self.yaw_list.clear()
                    self.frame_idx = 0

                    self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
                    self.last_cx_hoop = self.last_cy_hoop = None
                    self.last_z_hoop = self.last_yaw = None
                    self.last_box_hoop = None

                    self.cam2_miss_count = 0
                    self.hoop_lost = 0

                    self.last_band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)
                    self.line_start_time = time.time()

                    self.rect_color = (0, 255, 0)

                    self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames')
            
            if self.collecting:
                self.frame_idx += 1
                self.get_logger().info(f"step {self.frame_idx}")

                # 1. ROI HSV / Depth 준비
                self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

                depth_mask = np.zeros_like(roi_depth, dtype=np.uint8)
                depth_mask[(roi_depth > self.depth_min) & (roi_depth < self.depth_max_hoop)] = 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                depth_mask_closed = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)

                # 2. 색상 마스크 (빨강 + 흰색)
                red_mask1 = cv2.inRange(self.hsv, (self.red_h1_low, self.red_s_low, self.red_v_low),
                                                    (self.red_h1_high, 255, 255))
                red_mask2 = cv2.inRange(self.hsv, (self.red_h2_low, self.red_s_low, self.red_v_low),
                                                    (self.red_h2_high, 255, 255))
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                red_mask[depth_mask_closed == 0] = 0
                red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
                red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

                white_mask = cv2.inRange(self.hsv, (0, 0, self.white_v_low),
                                                    (180, self.white_s_high, 255))
                white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
                white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

                # 거리/색/크기 조건을 만족하는 빨강 컨투어 선택
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 300:
                        valid_contours.append(c)

                best_score = best_top_score = best_left_score = best_right_score = 0.5
                best_cnt_hoop = None
                best_box = None
                # best_box_dil = None
                best_cx_hoop = best_cy_hoop = best_depth_hoop = None
                best_band_mask = best_left_mask = best_right_mask = None
                best_left_src = best_right_src = None

                if valid_contours:
                    # 각 컨투어를 expand_px만큼 팽창 후, 겹치면 병합
                    cand_mask = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
                    for c in valid_contours:
                        cv2.drawContours(cand_mask, [c], -1, 255, thickness=cv2.FILLED)

                    k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (2*self.expand_px+1, 2*self.expand_px+1))
                    cand_dil = cv2.dilate(cand_mask, k_merge, iterations=1)

                    merged_contours, _ = cv2.findContours(cand_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 크기 조건 한 번 더 (병합 후 잔여 노이즈 제거)
                    merged_contours = [c for c in merged_contours
                                    if cv2.contourArea(c) >= self.backboard_area]

                    # 최소 외접사각형 구하고 expand_px만큼 축소
                    for cnt in merged_contours:
                        rect = cv2.minAreaRect(cnt)
                        (cx, cy), (w, h), ang = rect
                        if w < h:
                            w, h = h, w
                            ang += 90.0
                        if w <= 2*self.expand_px or h <= 2*self.expand_px:
                            continue

                        rect_shrunk = ((cx, cy), (w - 2*self.expand_px, h - 2*self.expand_px), ang)
                        box = cv2.boxPoints(rect_shrunk).astype(np.float32)

                        # “ㄷ” 모양 판단
                        # 사각형 네 구역 정의 (top, left, right, inner)
                        s = box.sum(axis=1)
                        diff = np.diff(box, axis=1).ravel()
                        tl = box[np.argmin(s)]
                        br = box[np.argmax(s)]
                        tr = box[np.argmin(diff)]
                        bl = box[np.argmax(diff)]
                        src = np.array([tl, tr, br, bl], dtype=np.float32)
                        dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
                        M_inv = cv2.getPerspectiveTransform(dst, src)

                        top_dst   = np.array([[0,0],[w,0],[w,self.band_top_ratio*h],[0,self.band_top_ratio*h]], dtype=np.float32)
                        left_dst  = np.array([[0,0],[self.band_side_ratio*w,0], [self.band_side_ratio*w,h],[0,h]], dtype=np.float32)
                        right_dst = np.array([[w - self.band_side_ratio*w,0],[w,0],[w,h], [w - self.band_side_ratio*w,h]], dtype=np.float32)
                        inner_dst = np.array([[self.band_side_ratio*w, self.band_top_ratio*h], [w - self.band_side_ratio*w, self.band_top_ratio*h],
                                            [w - self.band_side_ratio*w, h], [self.band_side_ratio*w, h]], dtype=np.float32)

                        def to_src(dst_poly):
                            return np.round(cv2.perspectiveTransform(dst_poly.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)

                        top_src, left_src, right_src, inner_src = map(to_src, [top_dst, left_dst, right_dst, inner_dst])

                        top_mask   = np.zeros_like(red_mask)
                        left_mask  = np.zeros_like(red_mask)
                        right_mask = np.zeros_like(red_mask)
                        inner_mask = np.zeros_like(red_mask)

                        cv2.fillPoly(top_mask,   [top_src], 255)
                        cv2.fillPoly(left_mask,  [left_src], 255)
                        cv2.fillPoly(right_mask, [right_src], 255)
                        cv2.fillPoly(inner_mask, [inner_src], 255)

                        area_top   = cv2.countNonZero(top_mask)
                        area_left  = cv2.countNonZero(left_mask)
                        area_right = cv2.countNonZero(right_mask)
                        area_inner = cv2.countNonZero(inner_mask)
                        if min(area_top, area_left, area_right, area_inner) < 5:
                            continue

                        ratio_top   = cv2.countNonZero(cv2.bitwise_and(red_mask, top_mask))   / area_top
                        ratio_left  = cv2.countNonZero(cv2.bitwise_and(red_mask, left_mask))  / area_left
                        ratio_right = cv2.countNonZero(cv2.bitwise_and(red_mask, right_mask)) / area_right
                        ratio_inner = cv2.countNonZero(cv2.bitwise_and(white_mask, inner_mask)) / area_inner

                        band_score = (ratio_top + ratio_left + ratio_right) / 3.0

                        if (ratio_top > self.red_ratio_min and
                            ratio_left > self.red_ratio_min and
                            ratio_right > self.red_ratio_min and
                            ratio_inner > self.white_inner_ratio_min):

                            inner_depth = roi_depth[inner_mask.astype(bool)]
                            hoop_valid  = np.isfinite(inner_depth) & (inner_depth > self.depth_min) & (inner_depth < self.depth_max_hoop)
                            if np.count_nonzero(hoop_valid) <= 10:
                                continue
                            depth_mean = float(np.mean(inner_depth[hoop_valid]))

                            if depth_mean < self.depth_min or depth_mean > self.depth_max_hoop:
                                continue

                            if best_score < band_score:
                                best_score = band_score
                                best_top_score, best_left_score, best_right_score = ratio_top, ratio_left, ratio_right
                                best_cnt_hoop = cnt
                                best_cx_hoop  = int(round(cx + self.roi_x_start))
                                best_cy_hoop  = int(round(cy + self.roi_y_start))
                                best_depth_hoop = depth_mean
                                best_box = (box + np.array([self.roi_x_start, self.roi_y_start])).astype(np.int32)
                                best_band_mask = top_mask + left_mask + right_mask
                                best_left_mask, best_right_mask = left_mask, right_mask
                                best_left_src, best_right_src = left_src, right_src

                # self.frame_idx += 1
                # self.get_logger().info(f"step {self.frame_idx}")
                
                # self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

                # depth_mask = np.zeros_like(roi_depth, dtype=np.uint8)
                # depth_mask[(roi_depth > self.depth_min) & (roi_depth < self.depth_max_hoop)] = 255

                # # 모폴로지 close 적용
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # depth_mask_closed = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)

                # # 빨강 마스킹
                # red_mask1 = cv2.inRange(self.hsv, (self.red_h1_low, self.red_s_low, self.red_v_low), (self.red_h1_high, 255, 255))
                # red_mask2 = cv2.inRange(self.hsv, (self.red_h2_low, self.red_s_low, self.red_v_low), (self.red_h2_high, 255, 255))
                # red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                # red_mask[depth_mask_closed == 0] = 0  

                # white_mask = cv2.inRange(self.hsv, (0, 0, self.white_v_low), (180, self.white_s_high, 255))

                # # 모폴로지
                # red_mask_raw = red_mask.copy()

                # cnts0, _ = cv2.findContours(red_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # red_mask_filtered0 = np.zeros_like(red_mask_raw, dtype=np.uint8)
                # for c in cnts0:
                #     if cv2.contourArea(c) >= 300:
                #         cv2.drawContours(red_mask_filtered0, [c], -1, 255, thickness=cv2.FILLED)

                # # 이제 모폴로지를 red_mask_filtered0에 적용
                # red_mask = cv2.morphologyEx(red_mask_filtered0, cv2.MORPH_OPEN,  kernel)
                # red_mask = cv2.morphologyEx(red_mask,         cv2.MORPH_CLOSE, kernel)

                # # (옵션) white도 동일하게 하고 싶으면 그대로 반복
                # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel)
                # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

                # # 컨투어
                # contours, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                
                # best_box = best_cnt_hoop = None
                # best_score = best_top_score = best_left_score = best_right_score = 0.5
                # best_cx_hoop = best_cy_hoop = best_depth_hoop = best_yaw = None
                # best_band_mask = best_left_mask = best_right_mask = best_left_src = best_right_src = None

                # dbg_dil_union = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)  

                # for cnt in contours:
                #     area = cv2.contourArea(cnt)
                #     if area > self.backboard_area: # 1. 일정 넓이 이상
                #         # 현재 cnt의 bbox 확장(이 범위에 겹치는 컨투어는 같은 후보로 간주)
                #         x, y, w0, h0 = cv2.boundingRect(cnt)
                #         x0 = max(0, x - self.prox_px)
                #         y0 = max(0, y - self.prox_px)
                #         x1 = min(self.roi_w-1, x + w0 + self.prox_px)
                #         y1 = min(self.roi_h-1, y + h0 + self.prox_px)
                #         expanded_rect = (x0, y0, x1-x0, y1-y0)

                #         def _overlap(r1, r2):
                #             ax, ay, aw, ah = r1
                #             bx, by, bw, bh = r2
                #             return not (ax+aw < bx or bx+bw < ax or ay+ah < by or by+bh < ay)

                #         min_piece_area = getattr(self, 'min_piece_area', int(0.15 * self.backboard_area))

                #         cand_mask = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)

                #         # 현재 cnt + 주변의 유의미한 조각들을 함께 그림
                #         for c2 in contours:
                #             if cv2.contourArea(c2) < min_piece_area:
                #                 continue
                #             xx, yy, ww, hh = cv2.boundingRect(c2)
                #             if _overlap(expanded_rect, (xx, yy, ww, hh)):
                #                 cv2.drawContours(cand_mask, [c2], -1, 255, thickness=cv2.FILLED)

                #         # 
                #         k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.expand_px+1, 2*self.expand_px+1))
                #         cand_dil = cv2.dilate(cand_mask, k_merge, iterations=1)

                #         cnts_dil, _ = cv2.findContours(cand_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #         if not cnts_dil:
                #             continue
                #         cnt_dil = max(cnts_dil, key=cv2.contourArea)


                #         for c_d in cnts_dil:
                #             cv2.drawContours(dbg_dil_union, [c_d], -1, 255, thickness=cv2.FILLED)

                #         # 4) 팽창된 컨투어의 최소외접사각형
                #         rect_dil = cv2.minAreaRect(cnt_dil)        # ((cx,cy),(w,h),ang)
                #         (cx, cy), (w, h), ang = rect_dil
                #         if w < h:
                #             w, h = h, w
                #             ang += 90.0

                #         # 5) 균일 침식(내측으로 shrink)  배경/가장자리 잔여 제거
                #         if w <= 2*self.shrink_px or h <= 2*self.shrink_px:
                #             continue
                #         w2 = w - 2*self.shrink_px
                #         h2 = h - 2*self.shrink_px
                #         rect_shrunk = ((cx, cy), (w2, h2), ang)

                #         # 6) shrunk-rect 폴리곤 (ROI 좌표계)
                #         box2 = cv2.boxPoints(rect_shrunk).astype(np.float32)

                #         # 꼭짓점 정렬 (tl,tr,br,bl)
                #         s2 = box2.sum(axis=1)
                #         diff2 = np.diff(box2, axis=1).ravel()
                #         tl2 = box2[np.argmin(s2)]
                #         br2 = box2[np.argmax(s2)]
                #         tr2 = box2[np.argmin(diff2)]
                #         bl2 = box2[np.argmax(diff2)]
                #         src2 = np.array([tl2, tr2, br2, bl2], dtype=np.float32)

                #         # 7) shrunk-rect 기준으로 밴드/내부 정의 (정규화 평면)
                #         dst_rect2 = np.array([[0,0],[w2,0],[w2,h2],[0,h2]], dtype=np.float32)

                #         # 실제 백보드 비율 유지
                #         top2  = max(1, int(round(h2 * self.band_top_ratio)))
                #         side2 = max(1, int(round(w2 * self.band_side_ratio)))

                #         top_dst2   = np.array([[0,0],[w2,0],[w2,top2],[0,top2]], dtype=np.float32)
                #         left_dst2  = np.array([[0.2*side2,0],[side2,0],[side2,h2],[0.2*side2,h2]], dtype=np.float32)
                #         right_dst2 = np.array([[w2-side2,0],[w2-0.2*side2,0],[w2-0.2*side2,h2],[w2-side2,h2]], dtype=np.float32)
                #         inner_dst2 = np.array([[side2,top2],[w2-side2,top2],[w2-side2,h2],[side2,h2]], dtype=np.float32)

                #         # 8) 변환행렬 (shrunk-rect ↔ 정규화 평면)
                #         M2_inv = cv2.getPerspectiveTransform(dst_rect2.astype(np.float32), src2.astype(np.float32))

                #         # 9) 최종 마스크(ROI 좌표계) 생성
                #         def to_src2(dst_poly):
                #             return np.round(
                #                 cv2.perspectiveTransform(dst_poly.reshape(-1,1,2), M2_inv).reshape(-1,2)
                #             ).astype(np.int32)

                #         top_src2   = to_src2(top_dst2)
                #         left_src2  = to_src2(left_dst2)
                #         right_src2 = to_src2(right_dst2)
                #         inner_src2 = to_src2(inner_dst2)

                #         top_mask   = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
                #         left_mask  = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
                #         right_mask = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
                #         inner_mask = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
                #         band_mask  = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)

                #         cv2.fillPoly(top_mask,   [top_src2], 255, lineType=cv2.LINE_8)
                #         cv2.fillPoly(left_mask,  [left_src2], 255, lineType=cv2.LINE_8)
                #         cv2.fillPoly(right_mask, [right_src2], 255, lineType=cv2.LINE_8)
                #         cv2.fillPoly(inner_mask, [inner_src2], 255, lineType=cv2.LINE_8)
                #         cv2.fillPoly(band_mask,  [top_src2, left_src2, right_src2], 255, lineType=cv2.LINE_8)

                #         area_top   = cv2.countNonZero(top_mask)
                #         area_left  = cv2.countNonZero(left_mask)
                #         area_right = cv2.countNonZero(right_mask)
                #         if area_top < 5 or area_left < 5 or area_right < 5:
                #             continue

                #         ratio_top   = cv2.countNonZero(cv2.bitwise_and(red_mask, top_mask))     / float(area_top)
                #         ratio_left  = cv2.countNonZero(cv2.bitwise_and(red_mask, left_mask))    / float(area_left)
                #         ratio_right = cv2.countNonZero(cv2.bitwise_and(red_mask, right_mask))   / float(area_right)
                #         ratio_band  = (ratio_top + ratio_left + ratio_right) / 3.0

                #         if ratio_top >= self.red_ratio_min and ratio_left >= self.red_ratio_min and ratio_right >= self.red_ratio_min:
                #             area_inner  = cv2.countNonZero(inner_mask)
                #             white_hits  = cv2.countNonZero(cv2.bitwise_and(white_mask, inner_mask))
                #             ratio_inner = white_hits / float(area_inner) if area_inner > 0 else 0.0
                #             if ratio_inner <= self.white_inner_ratio_min:
                #                 continue

                #             inner_depth = roi_depth[inner_mask.astype(bool)]
                #             hoop_valid  = np.isfinite(inner_depth) & (inner_depth > self.depth_min) & (inner_depth < self.depth_max_hoop)
                #             if np.count_nonzero(hoop_valid) <= 10:
                #                 continue
                #             depth_mean = float(np.mean(inner_depth[hoop_valid]))
                #             if depth_mean < self.depth_min or depth_mean > self.depth_max_hoop:
                #                 continue

                #             if best_score < ratio_band:
                #                 best_score = ratio_band
                #                 best_top_score, best_left_score, best_right_score = ratio_top, ratio_left, ratio_right
                #                 best_cnt_hoop = cnt
                #                 best_cx_hoop  = int(round(cx + self.roi_x_start))
                #                 best_cy_hoop  = int(round(cy + self.roi_y_start))
                #                 best_depth_hoop = depth_mean

                #                 draw_box = cv2.boxPoints(rect_shrunk).astype(np.float32)
                #                 best_box = (draw_box + np.array([self.roi_x_start, self.roi_y_start], dtype=np.float32)).astype(np.int32)

                #                 draw_rect_dil = cv2.boxPoints(rect_dil).astype(np.float32)
                #                 best_box_dil = (draw_rect_dil + np.array([self.roi_x_start, self.roi_y_start], dtype=np.float32)).astype(np.int32)

                #                 best_band_mask = band_mask
                #                 best_left_mask = left_mask
                #                 best_right_mask = right_mask
                #                 best_left_src  = left_src2
                #                 best_right_src = right_src2

                # 백보드 검출은 끝났고 각도 계산 및 정보 갱신
                if best_cnt_hoop is not None: # 골대 탐지를 했다면
                    self.hoop_lost = 0

                    self.last_score, self.last_top_score, self.last_left_score, self.last_right_score = best_score, best_top_score, best_left_score, best_right_score
                    self.last_cx_hoop, self.last_cy_hoop = best_cx_hoop, best_cy_hoop
                    self.last_z_hoop = best_depth_hoop
                    self.last_box_hoop = best_box
                    self.last_band_mask = best_band_mask

                    self.rect_color = (0, 255, 0)

                    left_vals = roi_depth[best_left_mask.astype(bool)] # 왼쪽 깊이
                    right_vals = roi_depth[best_right_mask.astype(bool)] # 오른쪽 깊이

                    left_valid = np.isfinite(left_vals) & (left_vals > self.depth_min) & (left_vals < self.depth_max_hoop)
                    right_valid = np.isfinite(right_vals) & (right_vals > self.depth_min) & (right_vals < self.depth_max_hoop)

                    nL = int(np.count_nonzero(left_valid))
                    nR = int(np.count_nonzero(right_valid)) # 잘 받아오고 있나 개수 검출

                    if nL >= 10 and nR >= 10:
                        depth_left = float(np.mean(left_vals[left_valid])) 
                        depth_right = float(np.mean(right_vals[right_valid])) 

                        x_left_px_roi = float(best_left_src[:, 0].mean())
                        x_right_px_roi = float(best_right_src[:, 0].mean())
                        x_left_px_full = x_left_px_roi + self.roi_x_start
                        x_right_px_full = x_right_px_roi + self.roi_x_start

                        x_left = (x_left_px_full  - self.cx_intr) * depth_left / self.fx
                        x_right = (x_right_px_full - self.cx_intr) * depth_right / self.fx # 거리 계산

                        dx = x_right - x_left
                        dz = depth_left - depth_right  # 오른쪽이 더 가까우면 + >>>> 내가 회전할 각도

                        if abs(dx) > 1:  # 1mm
                            best_yaw = math.atan2(dz, dx)
                            is_hoop_valid = True   # 어떤 값 쓸건지 판단용
                            self.last_yaw = best_yaw             

                            self.get_logger().info(f"[Hoop] Found! | Dist: {best_depth_hoop:.2f}mm | Pos: {best_cx_hoop}, {-best_cy_hoop}, " 
                                                   f"| Acc: {best_score:.2f}, | Ang: {math.degrees(best_yaw):.2f}")
                        else: # 그럴일 없다
                            best_yaw = 0
                            is_hoop_valid = False
                            self.get_logger().info(f"[HOOP] Retry | not enough big box")
                    else: # 깊이를 못 받아온 경우 >> 오류거나 반사가 너무 심하거나
                        self.get_logger().info(f"[HOOP] Retry | not enough valid depth")
                        best_yaw = 0
                        is_hoop_valid = False

                    self.last_yaw = best_yaw

                    cv2.polylines(frame, [best_box], True, self.rect_color, 2)
                    cv2.circle(frame, (best_cx_hoop, best_cy_hoop), 5, (0, 0, 255), 2)

                    # cv2.polylines(frame, [best_box_dil], True, (255, 255, 0), 1)

                    self.last_position_text = f'Dist: {best_depth_hoop:.2f}mm | Pos: {best_cx_hoop}, {-best_cy_hoop}, | Acc: {best_score:.2f}, | Ang: {math.degrees(best_yaw):.2f}'
                    self.backboard_score_text = f'Top: {best_top_score:.2f} | Left: {best_left_score:.2f} | Right: {best_right_score:.2f}'

                else:
                    if self.hoop_lost < 3 and self.last_box_hoop is not None:     ## .;;      
                        self.hoop_lost += 1

                        best_score, best_top_score, best_left_score, best_right_score = self.last_score, self.last_top_score, self.last_left_score, self.last_right_score
                        best_cx_hoop, best_cy_hoop = self.last_cx_hoop, self.last_cy_hoop
                        best_depth_hoop = self.last_z_hoop 
                        best_box = self.last_box_hoop
                        best_band_mask = self.last_band_mask
                        best_yaw = self.last_yaw

                        self.rect_color = (255, 0, 0)

                        is_hoop_valid = True

                        self.get_logger().info(f"[Hoop] Lost! | Dist: {best_depth_hoop:.2f}mm | Pos: {best_cx_hoop}, {-best_cy_hoop}, | Acc: {best_score:.2f}, | Ang: {math.degrees(best_yaw):.2f}")

                        cv2.polylines(frame, [best_box], True, self.rect_color, 2)
                        cv2.circle(frame, (best_cx_hoop, best_cy_hoop), 5, (0, 0, 255), 2)

                        self.last_position_text = f'Dist: {best_depth_hoop:.2f}mm | Pos: {best_cx_hoop}, {-best_cy_hoop}, | Acc: {best_score:.2f}, | Ang: {math.degrees(best_yaw):.2f}'
                        self.backboard_score_text = f'Top: {best_top_score:.2f} | Left: {best_left_score:.2f} | Right: {best_right_score:.2f}'

                    else:
                        self.hoop_lost = 3

                        best_cx_hoop = best_cy_hoop = best_depth_hoop = None # cy는 필요없긴 함
                        best_yaw = None

                        self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
                        self.last_cx_hoop = self.last_cy_hoop = None
                        self.last_z_hoop = self.last_yaw = None
                        self.last_box_hoop = None

                        self.rect_color = (0, 0, 255)

                        is_hoop_valid = False

                        self.get_logger().info(f"[Hoop] Miss!")
                        
                        self.last_position_text = f'Miss'
                        self.backboard_score_text = f"Miss"
                        self.last_band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)

                self.frames_left -= 1
                self.hoop_valid_list.append(is_hoop_valid)
                self.hoop_cx_list.append(best_cx_hoop)
                self.hoop_dis_list.append(best_depth_hoop)
                self.yaw_list.append(best_yaw)

                if self.frames_left <= 0:
                    result = Counter(self.hoop_valid_list).most_common(1)[0][0]

                    process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0 # 총 시간

                    if result == True: # 탐지가 더 많음
                        self.hoop_miss_count = 0 
                        cxs = [a for s, a in zip(self.hoop_valid_list, self.hoop_cx_list) if s == result and a is not None]
                        dists = [a for s, a in zip(self.hoop_valid_list, self.hoop_dis_list) if s == result and a is not None]
                        yaws = [a for s, a in zip(self.hoop_valid_list, self.yaw_list) if s == result and a is not None]

                        avg_cx = int(round(np.mean(cxs)))
                        avg_dis = round(np.mean(dists),2)
                        avg_yaw_rad = np.median(yaws)
                        avg_yaw_deg = round(math.degrees(avg_yaw_rad),2)

                        self.last_avg_cx_hoop = avg_cx # 놓쳤을 때 판단용
                        self.depth_max_hoop = avg_dis + 300 # 최대거리 제한

                        if 1200 < avg_dis: # 거리가 1.2m 이상 > 일단 골대 쪽으로 직진하기

                            angle = int(round(math.degrees(math.atan((avg_cx - self.cx_intr) / self.fx))))
                            
                            if angle >= 12:
                                res = 30
                            elif angle <= -12:
                                res = 29
                            else:
                                res = 1
                            self.get_logger().info(f"[Hoop] Approaching | x: {avg_cx}, dis: {avg_dis}, "
                                                f"line angle= {angle}, backboard angle= {avg_yaw_deg}")
                            
                        elif 800 < avg_dis: # 거리가 0.8m 이상 > 일단 골대 쪽으로 직진하기

                            angle = int(round(math.degrees(math.atan((avg_cx - self.cx_intr) / self.fx))))
                            
                            if angle >= 12:
                                res = 30
                            elif angle <= -12:
                                res = 29
                            else:
                                res = 27
                            self.get_logger().info(f"[Hoop] Approaching | x: {avg_cx}, dis: {avg_dis}, "
                                                f"line angle= {angle}, backboard angle= {avg_yaw_deg}")
                        
                        else: # 거리가 0.7m 이내 > 
                            self.hoop_near_by = True # 타원 고?
                            angle = 0
                            beta_rad = math.atan((avg_cx - self.cx_intr) / self.fx) # 백보드 점과의 각도
                            x_ = avg_dis * math.sin(beta_rad) + self.hoop_r * math.sin(avg_yaw_rad)
                            z_ = avg_dis * math.cos(beta_rad) - self.hoop_r * math.cos(avg_yaw_rad) - self.throwing_range_z # x_, z_ 내 기준 링 중심 좌표 mm

                            x_2 = x_ - 2 * self.hoop_r * math.sin(avg_yaw_rad) # 백보드 맞추고
                            z_2 = z_ + 2 * self.hoop_r * math.cos(avg_yaw_rad)  

                            self.last_position_text = f'x: {x_}, z: {z_}, alpha = {avg_yaw_deg}, beta= {math.degrees(beta_rad)}, x2: {x_2}, z2: {z_2}'

                            if math.hypot(x_,z_) < self.hoop_r or math.hypot(x_2, z_2) < self.hoop_r:
                                res = 17
                                self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
                                self.last_cx_hoop = self.last_cy_hoop = self.last_z_hoop = self.last_yaw = None
                                self.last_box_hoop = None
                                self.last_band_mask = np.zeros((self.roi_y_end - self.roi_y_start, self.roi_x_end - self.roi_x_start), dtype=np.uint8)  

                                self.hoop_near_by = False
                                self.last_avg_cx_hoop = self.zandi_x
                                self.cam1_mode = BALL
                                self.backboard_score_text = "Ball Now"
                                self.depth_max_hoop = 1500
                                self.apply_mode_layout()

                            elif z_ <= -25: 
                                res = 5 # 뒤로가기

                            elif abs(x_) >= z_:
                                if x_ > 0:
                                    if x_ > 300:
                                        res = 3
                                    else:   
                                        res = 8 # 오른
                                else: 
                                    if x_ < -300:
                                        res = 2
                                    else:
                                        res = 7 # 왼
                            else:
                                res = 12 # 앞
                                

                            if res == 17:
                                self.get_logger().info(f"[Hoop] Shoot !! | x: {avg_cx}, dis: {avg_dis}, "
                                                f"backboard angle= {avg_yaw_deg}")  

                            else: 
                                self.get_logger().info(f"[Hoop] Almost near by | x: {avg_cx}, dis: {avg_dis}, "
                                                f"backboard angle= {avg_yaw_deg} , beta= {math.degrees(beta_rad)}")

                    else:
                        if self.hoop_near_by:
                            self.depth_max_hoop += 100
                            self.hoop_miss_count += 1
                            if self.hoop_miss_count <= 2:
                                if self.last_avg_cx_hoop - self.zandi_x < -150:
                                    res = 2 # 왼
                                    angle = -10
                                elif self.last_avg_cx_hoop - self.zandi_x > 150:
                                    res = 3 # 오
                                    angle = 10
                                else:
                                    res = 5 # 뒤
                                    angle = 0

                                self.get_logger().info(f"[Hoop] Try to find hoop,,, | miss= {self.hoop_miss_count}")
                                self.last_position_text = "No hoop, try to find,,,"#######################################

                            elif self.hoop_miss_count <= 6: # 근접했는데 놓쳤으면 뒤로가서 확인해봐라
                                res = 5 # 다시 찾는 모션 >> 제자리에서? 한발짝 뒤에서?
                                angle = 0

                                self.get_logger().info(f"[Hoop] Try again,,, | miss= {self.hoop_miss_count}")
                                self.last_position_text = "No hoop"

                            else: # GG요 (3번 이상 놓쳤을 때)
                                self.get_logger().info(f"[Hoop] No hoop detected")
                                self.last_position_text = "[Hoop] No hoop, "

                                res = 99 # 라인 걸으세요
                                angle = 0
                                self.hoop_near_by = False
                                self.last_avg_cx_hoop = self.zandi_x
                                self.hoop_miss_count = 0
                                self.depth_max_hoop = 2000
                        
                        else:
                            self.get_logger().info(f"[Hoop] No hoop detected")
                            self.last_position_text = "[Hoop] No hoop"

                            res = 99 # 라인 걸으세요
                            angle = 0
                            self.last_avg_cx_hoop = self.zandi_x
                            self.hoop_miss_count = 0

                    # 퍼블리시
                    msg_out = BallResult()
                    msg_out.res = res
                    msg_out.angle = abs(angle)
                    self.ball_result_pub.publish(msg_out)

                    self.get_logger().info(f"res= {res}")
                    self.get_logger().info(f"frames= {len(self.hoop_valid_list)}, wall= {process_time*1000:.1f} ms")

                    # 리셋
                    self.collecting = False
                    self.frames_left = 0
                    self.frame_idx = 0

            cv2.rectangle(frame, (int((self.roi_x_start + self.roi_x_end) / 2) - 50, self.roi_y_start), 
                (int((self.roi_x_start + self.roi_x_end) / 2) + 50, self.roi_y_end), (255, 120, 150), 1)

        cv2.rectangle(frame, (self.roi_x_start + 1, self.roi_y_start + 1), (self.roi_x_end - 1, self.roi_y_end - 1), self.rect_color, 1)

        # 딜레이 측정
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        now = time.time()
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            fps = self.frame_count / (now - self.last_report_time)
            self.last_avg_text = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        cv2.putText(frame, self.backboard_score_text, (10, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)

        if self.collecting:
            if self.cam1_mode == BALL:
                cv2.imshow('Basketball Mask', mask) 
                if len(contours_large) != 0:
                    cv2.imshow('Expanded Mask', expanded_mask)
                    cv2.imshow('Exp11nded Mask', second_mask)
            elif self.cam1_mode == HOOP:
                cv2.imshow('Red Mask', red_mask)
                cv2.imshow('White Mask', white_mask)
                cv2.imshow('Depth Ma1sk', depth_mask)
                cv2.imshow('Depth Mask', depth_mask_closed)
                if best_band_mask is not None:
                    cv2.imshow('Band Mask', self.last_band_mask)

        cv2.imshow('Detection', frame)
        cv2.waitKey(1)

    def decide_to_pick(self, dx, dy, process_time):
        if abs(dx) <=27 and abs(dy) <= 18: # 오케이 조준 완료  310 360     315 285
            res = 9 # pick 모션
            self.picked = True
            self.pick_attempt += 1
            self.get_logger().info(f"[Ball] Pick! | Pos : {dx}, {-dy}, attempt {self.pick_attempt}")
            self.last_position_text = "[Ball] Pick!"

        
        else: # x합격, y 합격 나누고 둘다 불합격일떄만 크기 비교해서 하기
            self.get_logger().info(f"[Ball] Found! | Pos : {dx}, {-dy}, attempt {self.pick_attempt}")
            self.last_position_text = f"[Ball] Position: {dx}, {-dy}"

            if dy > 40: # 일단 y로 너무 붙으면 뒤로 가기
                res = 5
                self.get_logger().info(f"[Ball] 111111111111")
            else: 
                self.get_logger().info(f"[Ball] 222222222222")
                if abs(dx) <= 27 and abs(dy) > 18:
                    if abs(dy) >= 60:
                        if dy > 0:
                            res = 5 #back_one

                        elif dy < 0:
                            res = 12 #forward_one
                    else:
                        if dy > 0:
                            res = 5 #back_half

                        elif dy < 0:
                            res = 6 #forward_half

                elif abs(dx) > 27 and abs(dy) <= 18:
                    if dx > 0: 
                        res = 8 #right_half
                
                    elif dx < 0:
                        res = 7 #left_half
                
                elif abs(dx) > 27 and abs(dy) > 18:
                    if abs(dx) >= abs(dy):
                        if dx > 0: 
                            res = 8 #right_half
                    
                        elif dx < 0:
                            res = 7 #left_half

                    elif abs(dx) < abs(dy):
                        if abs(dy) >= 60:
                            if dy > 0:
                                res = 5 #back_one

                            elif dy < 0:
                                res = 12 #forward_one
                        else:
                            if dy > 0:
                                res = 5 #back_half

                            elif dy < 0:
                                res = 6 #forward_half

                else: # 여기로 빠질 일 없음
                    self.get_logger().info(f"[Ball] CAM2 Found, Relative position: {dx}, {-dy} | "
                                    f"frames= {len(self.ball_valid_list)}, "
                                    f"wall= {process_time*1000:.1f} ms")
                    res = 5 
        return res

    def cam2_image_callback(self, cam2_color_msg: Image): # cam1, cam2 분리하기

        if self.cam_mode != CAM2:
            return
        
        start_time = time.time()
        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(cam2_color_msg, desired_encoding='bgr8')
        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]


        if not self.collecting:
            if self.armed:
                self.collecting = True
                self.armed = False
                self.frames_left = self.collecting_frames
                self.window_id += 1 

                # 초기화
                self.ball_valid_list.clear()
                self.ball_cx_list.clear()
                self.ball_cy_list.clear()
                self.ball_dis_list.clear()
                self.frame_idx = 0

                self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                self.ball_lost = 0
                self.cam1_ball_count = 0
                
                self.line_start_time = time.time()
                
                self.draw_color = (0, 255, 0)
                self.rect_color = (0, 255, 0)

                self.get_logger().info(f'[Star121212121212t] Window {self.window_id} | I got {self.collecting_frames} frames in CAM{self.cam_mode}')
        
        if self.collecting:
            
            angle = 0
            res=0
            self.frame_idx += 1
            self.get_logger().info(f"step {self.frame_idx}")
            
            # HSV 색 조절
            self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            raw_mask = cv2.inRange(self.hsv, self.lower_hsv_ball2, self.upper_hsv_ball2) # 주황색 범위 색만
            
            # 모폴로지 연산
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어

            best_cnt_ball = None
            best_ratio_ball = 0.3
            best_cx_ball = best_cy_ball = best_radius = None

            for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
                area = cv2.contourArea(cnt) # 1. 면적 > 1000
                if area > 700:
                    (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                    circle_area = circle_r * circle_r * math.pi
                    ratio = area / circle_area
                    # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈1
                    if ratio > best_ratio_ball:
                        best_cnt_ball = cnt
                        best_ratio_ball = ratio
                        best_cx_ball = int(x + self.roi_x_start)
                        best_cy_ball = int(y + self.roi_y_start)
                        best_radius = int(circle_r)

            # 검출 결과 처리: 이전 위치 유지 로직
            if best_cnt_ball is not None:
                self.ball_lost = 0
                
                self.last_cx_ball = best_cx_ball
                self.last_cy_ball = best_cy_ball
                self.last_radius = best_radius

                self.rect_color = (0, 255, 0)
                self.draw_color = (255, 0, 0)
                is_ball_valid = True

                self.get_logger().info(f"[Ball] Found! | {best_cx_ball}, {best_cy_ball}")

                cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)

            else:
                if self.ball_lost < 3 and self.last_cx_ball is not None:
                    self.ball_lost += 1

                    best_cx_ball = self.last_cx_ball
                    best_cy_ball = self.last_cy_ball
                    best_radius = self.last_radius

                    self.rect_color = (255, 0, 0)
                    self.draw_color = (0, 255, 255)
                    is_ball_valid = True

                    self.get_logger().info(f"[Ball] Lost! | {best_cx_ball}, {best_cy_ball}")

                    cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)
                else:
                    self.ball_lost = 3

                    best_cx_ball = best_cy_ball = best_radius = None
                    self.last_cx_ball = None
                    self.last_cy_ball = None
                    self.last_radius = None

                    self.rect_color = (0, 0, 255)
                    is_ball_valid = False

                    self.get_logger().info(f"[Ball] Miss!")
            
            self.frames_left -= 1
            self.ball_valid_list.append(is_ball_valid)
            self.ball_cx_list.append(best_cx_ball)
            self.ball_cy_list.append(best_cy_ball)

            if self.frames_left <= 0:
                result = self.ball_valid_list.count(True) >= 5

                process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                if result == True: # 공을 봤다면
                    if self.ball_never_seen: # 여긴 그냥 지금껏 공을 본 적 있는지 없는지 저장용
                        self.ball_saw_once = True
                        self.ball_never_seen = False

                    self.cam2_miss_count = 0

                    cxs = [a for s, a in zip(self.ball_valid_list, self.ball_cx_list) if s == result and a is not None]
                    cys = [a for s, a in zip(self.ball_valid_list, self.ball_cy_list) if s == result and a is not None]
                    avg_cx = int(round(np.mean(cxs)))
                    avg_cy = int(round(np.mean(cys)))

                    dx = avg_cx - self.pick_x
                    dy = avg_cy - self.pick_y
                    self.last_avg_cx_ball = avg_cx
                    self.last_avg_cy_ball = avg_cy

                    if self.picked: # 줍기 모션을 했다면 >> 나 방금 주웠는데 눈 앞에 공이 또 있네?
                        if self.pick_attempt >= 3: # 3번 넘게 실패
                            res = 22
                            self.get_logger().info(f"[Ball] Failed to pick,,, I'll give up | Pos : {dx}, {-dy} | attempt= {self.pick_attempt}")

                            self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                            self.pick_attempt = 0
                            self.last_position_text = "[Hoop] Hoop Now"

                            self.picked = False
                            self.ball_saw_once = False
                            self.ball_never_seen = True
                            self.cam_mode = CAM1
                            self.cam1_mode = BALL # 공 못 주웠으니, 골대 찾을 필요가 없다 
                            self.apply_mode_layout()

                        else: # 시도는 3번만
                            res = self.decide_to_pick(dx, dy, process_time)
                            self.picked = True
                            self.get_logger().info(f"[Ball] Pick one more time!")

                    else: # 아직 줍지 않았고, 공은 발견함 (평소 루프)

                        # if math.hypot(dx, dy) <= self.pick_rad: # 오케이 조준 완료
                        #     self.cam_mode = CAM1
                        #     self.cam1_mode = HOOP

                        #     self.get_logger().info(f"[Ball] Pick! Pos : {dx}, {-dy} | "
                        #                     f"frames= {len(self.ball_valid_list)}, "
                        #                     f"wall= {process_time*1000:.1f} ms")
                        #     self.get_logger().info(f"[Hoop] Findind hoop,,,")
                        #     res = 9 # pick 모션

                        #     self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                        #     self.backboard_score_text = "Hoop Now"
                        res = self.decide_to_pick(dx, dy, process_time)

                else: # 공을 못 봤다면
                    self.get_logger().info(f"picked = {self.picked}")
                    if self.picked: # 방금 줍는 모션을 했는데 내 눈앞에 공이 없다 = 잘 주웠다 (아니면 흘렸거나)
                        self.get_logger().info(f"[Ball] I made it! | , attempt {self.pick_attempt}")
                        res = 10 # 재그립 모션

                        self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                        self.backboard_score_text = ""
                        self.last_position_text = f"[Hoop] Hoop Now"

                        self.cam2_miss_count = 0
                        self.pick_attempt = 0
                        self.picked = False
                        self.ball_saw_once = False
                        self.ball_never_seen = True

                        self.cam_mode = CAM1
                        self.cam1_mode = HOOP # 골대 찾으러 가자
                        self.apply_mode_layout()


                    else: # 공을 탐지도 못 했고, 줍지도 않았다
                        if self.ball_saw_once: # 근데 그 전에 공을 발견한 적이 있긴 함
                            # self.cam2_miss_count += 1
                            res = 99
                            
                            # if self.cam2_miss_count <= 3: # 한번만 다시 봐봐
                            #     res = 99
                                
                            #     self.get_logger().info(f"[Ball] Retry,,, | miss= {self.cam2_miss_count}")
                            #     self.last_position_text = "[Ball] Miss"
                            #     # 오른쪽 움직일 때 회전하느라 공 뒤로 사라지는 거 생각하기

                            # elif self.cam2_miss_count <= 5:
                            #     self.get_logger().info(f"[Ball] Retry,,, | miss= {self.cam2_miss_count}")
                            #     res = 5
                            #     self.last_position_text = "[Ball] Miss"

                            # else: # 한 번 봐놓고 공을 5번 연속이나 못 보면 뭐할까?  >>>  뭔가 찾는 모션
                            #     self.get_logger().info(f"[Ball] I totally missed,,,")
                            #     res = 99 # 라인 걸으세요
                            #     self.last_position_text = f""
                        
                            #     self.cam2_miss_count = 0
                            #     self.pick_attempt = 0
                            #     self.picked = False
                            #     self.ball_saw_once = False
                            #     self.ball_never_seen = True

                            #     self.cam_mode = CAM1
                            #     self.cam1_mode = BALL # 공 찾으러 가자
                            #     self.apply_mode_layout()
                        
                        else: # 공을 탐지도 못 했고, 줍지도 않았고 공을 본 적도 없음 = 방금 막 HOOP모드에서 바뀜 >> 직진만 하면서 찾아보자
                            self.get_logger().info(f"[Ball] Finding,,,")
                            res = 12
                            self.last_position_text = f"[Ball] finding a ball"
        
                msg_out = BallResult()
                msg_out.res = res
                msg_out.angle = angle
                self.last_res = res
                self.ball_result_pub.publish(msg_out)
                self.get_logger().info(f"res = {res}")
                self.get_logger().info(f"frames= {len(self.ball_valid_list)}, wall= {process_time*1000:.1f} ms")
                # 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0

        # else:
        #     self.last_position_text = "In move"

        cv2.rectangle(frame, (self.roi_x_start+1, self.roi_y_start+1), (self.roi_x_end-1, self.roi_y_end-1), self.rect_color, 1)
        cv2.rectangle(frame, (self.pick_x - 30, self.pick_y - 20), (self.pick_x + 30, self.pick_y + 20), (111,255,111), 1)
        # 딜레이 측정
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        now = time.time()
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            fps = self.frame_count / (now - self.last_report_time)
            self.last_avg_text = f'AVG: {avg_time * 1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        cv2.putText(frame, self.backboard_score_text, (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)

        if self.collecting:
            cv2.imshow('Basketball Mask', mask) # 기준 거리 이내, 주황색
        cv2.imshow('Ball Detection', frame)
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()