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

roi_x_start = int(camera_width * 0 // 5)
roi_x_end = int(camera_width * 5 // 5)
roi_y_start = int(camera_height * 1 // 12)
roi_y_end = int(camera_height * 11 // 12)

zandi_x = int((roi_x_start + roi_x_end) / 2)
zandi_y = int((roi_y_start + roi_y_end) / 2)

pick_x = zandi_x - 5
pick_y = zandi_y + 47 

class LineListenerNode(Node): #################################################################### 판단 프레임 수 바꿀 때 yolo_cpp 도 고려해라~~~
    def __init__(self):
        super().__init__('line_subscriber')

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)
        
        self.last_avg_text = "AVG: --- ms | FPS: --"
        self.last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력
        self.backboard_score_text = 'Miss' # H

        # 추적
        self.last_cx_ball = self.last_cy_ball = self.last_z_ball = self.last_radius = None
        self.last_agv_cy_ball = 0 # 화면 전환용
        self.ball_lost = 0 # B

        self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
        self.last_cx_hoop = self.last_cy_hoop = self.last_z_hoop = self.last_yaw = None
        self.last_box_hoop = None
        self.hoop_lost = 0 # H

        self.hoop_r = 0.1 # 골대 반지름
        self.throwing_range = 0.4 # 공 던지는 거리
        self.goal_range = 0.05 # 공 던질 때 떨어지는 곳 오차범위

        # 변수
        self.draw_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.fx, self.fy = 607.0, 606.0   # ros2 topic echo /camera/color/camera_info - 카메라 고유값
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.last_band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype = np.uint8)
        
        self.depth_max_ball = 1500.0  # mm 기준 마스크 거리
        self.depth_max_hoop = 2000.0
        self.depth_min = 50.0 
        self.depth_scale = 0.001    # m 변환용 >> B
        
        self.collecting = False     # 수집 중 여부
        self.armed = False               # motion_end 방어~!

        self.window_id = 0
        self.frame_idx = 0       
        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        self.collecting_frames = 15
        
        self.cam1_ball_count = 0
        self.cam2_miss_count = 0
        self.hoop_count = 0
        self.pick_attempt = 0

        self.picked = False
        self.ball_saw_once = False
        self.ball_never_seen = True
        self.hoop_near_by = False

        self.ball_valid_list = []
        self.ball_cx_list = []
        self.ball_cy_list = []
        self.ball_dis_list = [] # B

        self.hoop_valid_list = [] # H
        self.hoop_cx_list = []
        self.hoop_dis_list = []
        self.yaw_list = []

        self.bridge = CvBridge()   
                              
        self.cam1_color_sub = Subscriber(self, Image, '/cam1/color/image_raw') # CAM1
        self.cam1_depth_sub = Subscriber(self, Image, '/cam1/aligned_depth_to_color/image_raw') 
        self.cam1_sync = ApproximateTimeSynchronizer([self.cam1_color_sub, self.cam1_depth_sub], queue_size=5, slop=0.1)
        self.cam1_sync.registerCallback(self.cam1_image_callback)

        self.cam2_color_sub = self.create_subscription(  # CAM2
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

        self.declare_parameter("orange_h_low", 0) 
        self.declare_parameter("orange_h_high", 60)  
        self.declare_parameter("orange_s_low", 40) # 채도  
        self.declare_parameter("orange_s_high", 255)  
        self.declare_parameter("orange_v_low", 60) # 밝기
        self.declare_parameter("orange_v_high", 255)

        # 파라미터 선언 H
        self.declare_parameter('red_h1_low', 0) # 빨강
        self.declare_parameter('red_h1_high', 10)
        self.declare_parameter('red_h2_low', 160)   
        self.declare_parameter('red_h2_high', 180)
        self.declare_parameter('red_s_low', 80)
        self.declare_parameter('red_v_low', 60)
        
        self.declare_parameter('white_s_high', 130) # 하양 70. 190
        self.declare_parameter('white_v_low', 130)

        self.declare_parameter('band_top_ratio', 0.15)   # 백보드 h x 0.15
        self.declare_parameter('band_side_ratio', 0.10)   # w x 0.10

        self.declare_parameter('red_ratio_min', 0.55)     # 백보드 영역
        self.declare_parameter('white_min_inner', 0.50)  
        self.declare_parameter('backboard_area', 1500)

        # 파라미터 적용 B
        self.cam_mode = self.get_parameter("cam_mode").value
        self.cam1_mode = self.get_parameter("cam1_mode").value

        self.orange_h_low = self.get_parameter("orange_h_low").value
        self.orange_h_high = self.get_parameter("orange_h_high").value
        self.orange_s_low = self.get_parameter("orange_s_low").value
        self.orange_s_high = self.get_parameter("orange_s_high").value
        self.orange_v_low = self.get_parameter("orange_v_low").value
        self.orange_v_high = self.get_parameter("orange_v_high").value

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
        self.white_min_inner = self.get_parameter('white_min_inner').value
        self.backboard_area = self.get_parameter('backboard_area').value

        self.add_on_set_parameters_callback(self.param_callback)

        self.lower_hsv_ball = np.array([self.orange_h_low, self.orange_s_low, self.orange_v_low ], dtype=np.uint8) # 색공간 미리 선언
        self.upper_hsv_ball = np.array([self.orange_h_high, self.orange_s_high, self.orange_v_high], dtype=np.uint8)
        self.hsv = None 

        # 클릭
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ball Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Detection', self.on_click)
        cv2.setMouseCallback('Ball Detection', self.on_click)

    def param_callback(self, params):
        for p in params:
            if p.name == "cam_mode":   self.cam_mode = int(p.value)
            elif p.name == "cam1_mode": self.cam1_mode = int(p.value)
            elif p.name == "orange_h_low":   self.orange_h_low = int(p.value)
            elif p.name == "orange_h_high": self.orange_h_high = int(p.value)
            elif p.name == "orange_s_low":  self.orange_s_low = int(p.value)
            elif p.name == "orange_s_high": self.orange_s_high = int(p.value)
            elif p.name == "orange_v_low":  self.orange_v_low = int(p.value)
            elif p.name == "orange_v_high": self.orange_v_high = int(p.value)
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
            elif p.name == "white_min_inner": self.white_min_inner = float(p.value)
            elif p.name == "backboard_area":  self.backboard_area = int(p.value)

        # HSV 경계값 배열 갱신
        self.lower_hsv_ball = np.array([self.orange_h_low, self.orange_s_low, self.orange_v_low], dtype=np.uint8)
        self.upper_hsv_ball = np.array([self.orange_h_high, self.orange_s_high, self.orange_v_high], dtype=np.uint8)

        return SetParametersResult(successful=True)
        
    def on_click(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        if not (roi_x_start <= x <= roi_x_end and roi_y_start <= y <= roi_y_end):
            return

        rx = x - roi_x_start
        ry = y - roi_y_start
        k = 1  # (3,3)

        y0, y1 = max(0, ry - k), min(roi_y_end - roi_y_start, ry + k + 1)
        x0, x1 = max(0, rx - k), min(roi_x_end - roi_x_start, rx + k +1)
        patch = self.hsv[y0:y1, x0:x1].reshape(-1,3)

        H, S, V = np.mean(patch, axis=0).astype(int)
        self.get_logger().info(f"[Pos] x={x - zandi_x}, y={-(y - zandi_y)} | HSV=({H},{S},{V})")

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

        roi_color = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_depth = depth[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

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

                # HSV 색 조절
                self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
                raw_mask = cv2.inRange(self.hsv, self.lower_hsv_ball, self.upper_hsv_ball) # 주황색 범위 색만
                raw_mask[roi_depth >= self.depth_max_ball] = 0  
                raw_mask[roi_depth <= self.depth_min] = 0 

                # 모폴로지 연산
                mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

                # 컨투어
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어

                best_cnt_ball = None
                best_ratio_ball = 0.3
                best_cx_ball = best_cy_ball = best_z_ball = 0

                for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
                    area = cv2.contourArea(cnt) # 1. 면적 > 1000
                    if area > 500:
                        (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                        circle_area = circle_r * circle_r * math.pi
                        ratio = abs((area / circle_area) - 1)
                        # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈
                        if ratio < best_ratio_ball:
                            best_cnt_ball = cnt
                            best_ratio_ball = ratio
                            best_cx_ball = int(x + roi_x_start) 
                            best_cy_ball = int(y + roi_y_start)
                            best_radius = int(circle_r)

                # 검출 결과 처리: 이전 위치 유지 로직
                if best_cnt_ball is not None:
                    # 원 탐지를 했다면
                    self.ball_lost = 0
                    
                    x1 = max(best_cx_ball - roi_x_start - 1, 0)
                    x2 = min(best_cx_ball - roi_x_start + 2, roi_x_end - roi_x_start)
                    y1 = max(best_cy_ball - roi_y_start - 1, 0)
                    y2 = min(best_cy_ball - roi_y_start + 2, roi_y_end - roi_y_start)

                    roi_patch = roi_depth[y1:y2, x1:x2]
                    valid = np.isfinite(roi_patch) & (roi_patch > self.depth_min) & (roi_patch < self.depth_max_ball)
                    if np.count_nonzero(valid) >= 1:
                        best_z_ball = float(np.mean(roi_patch[valid])) * self.depth_scale
                        is_ball_valid = True
                        self.get_logger().info(f"Found! {best_cx_ball}, {best_cy_ball}, dist = {best_z_ball}")

                        self.last_z_ball = best_z_ball
                    else:
                        best_z_ball = -1 
                        is_ball_valid = False
                        self.get_logger().info(f"Invalid depth! {best_cx_ball}, {best_cy_ball}")
                    
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
                        self.get_logger().info(f"Lost! {best_cx_ball}, {best_cy_ball}, dist = {best_z_ball}")

                        cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)
                    else:
                        self.ball_lost = 3

                        best_cx_ball = best_cy_ball = best_z_ball = None
                        self.last_cx_ball = self.last_cy_ball = self.last_radius = self.last_z_ball = None
                        
                        self.rect_color = (0, 0, 255)

                        is_ball_valid = False
                        self.get_logger().info(f"Miss!")

                self.frames_left -= 1
                self.ball_valid_list.append(is_ball_valid)
                self.ball_cx_list.append(best_cx_ball)
                self.ball_cy_list.append(best_cy_ball)
                self.ball_dis_list.append(best_z_ball)

                if self.frames_left <= 0:
                    result = Counter(self.ball_valid_list).most_common(1)[0][0]

                    process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                    if result == True:
                        cxs = [a for s, a in zip(self.ball_valid_list, self.ball_cx_list) if s == result and a is not None]
                        cys = [a for s, a in zip(self.ball_valid_list, self.ball_cy_list) if s == result and a is not None]
                        dists = [a for s, a in zip(self.ball_valid_list, self.ball_dis_list) if s == result and a is not None]
                        avg_cx = int(round(np.mean(cxs)))
                        avg_cy = int(round(np.mean(cys)))
                        avg_dis = np.mean(dists)

                        angle = int(round(math.degrees(math.atan2(avg_cx - zandi_x, -avg_cy + 380))))
                        self.last_agv_cy_ball = avg_cy # 다음 프레임에 판단용

                        if angle > 90:
                            angle -= 180
                        elif angle <= -90:
                            angle += 180
                        
                        if angle >= 8:
                            res = 13 # 우회전
                        elif angle <= -8: # 좌회전
                            res = 14
                        else: # 직진
                            res = 12

                        self.get_logger().info(f"[Ball] Done: CAM1 found ball | {avg_cx}, {avg_cy}, dis: {avg_dis:.2f}, "
                                            f"angle= {angle}")
                        self.last_position_text = f"[Ball] Position: {avg_cx}, {avg_cy}"
                        self.cam1_ball_count += 1

                    else:
                        if self.last_agv_cy_ball >= 300 and self.cam1_ball_count >= 1: # 그 전까지 공을 보고 있었다면
                            res = 12
                            angle = 0
                            self.cam_mode = CAM2 # 2번 캠으로 ㄱㄱ
                            self.last_agv_cy_ball = 0

                            self.get_logger().info(f"[Ball] CAM1 Missed, CAM2 will find,,,")
                            
                            self.last_position_text = "Recieving,,,"
                            self.cam1_ball_count = 0

                            self.last_cx_ball = self.last_cy_ball = self.last_z_ball = self.last_radius = None
                            self.backboard_score_text = "Ball Now"

                        elif self.cam1_ball_count == 1:
                            self.get_logger().info(f"[Ball] Wrong thing")
                            res = 5 # 임시
                            angle = 0

                            self.last_position_text = "Back to line,,,"
                            self.cam1_ball_count = 0
                        
                        else: 
                            self.get_logger().info(f"[Ball] No ball detected")
                            res = 5 # 임시
                            angle = 0
                            
                            self.last_position_text = "No ball"
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
            else:
                self.last_position_text = "In move"

            cv2.line(frame, (roi_x_start, 280), (roi_x_end, 280), (250, 122, 122), 1)

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
                    self.last_band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)

                    self.hoop_lost = 0

                    self.line_start_time = time.time()

                    self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames')
            
            if self.collecting:
                self.frame_idx += 1
                self.get_logger().info(f"step {self.frame_idx}")
            
                t1 = time.time()
                
                self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

                t2 = time.time()

                # 빨강 마스킹
                red_mask1 = cv2.inRange(self.hsv, (self.red_h1_low, self.red_s_low, self.red_v_low), (self.red_h1_high, 255, 255))
                red_mask2 = cv2.inRange(self.hsv, (self.red_h2_low, self.red_s_low, self.red_v_low), (self.red_h2_high, 255, 255))
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                red_mask[roi_depth >= self.depth_max_hoop] = 0  
                red_mask[roi_depth <= self.depth_min] = 0 

                white_mask = cv2.inRange(self.hsv, (0, 0, self.white_v_low), (180, self.white_s_high, 255))
                
                t3 = time.time()

                # 모폴로지
                red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  self.kernel)
                red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel)

                white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  self.kernel)
                white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, self.kernel)

                t4 = time.time()

                # 컨투어
                contours, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 출력용 카피
                
                best_box = best_cnt_hoop = None
                best_score = best_top_score = best_left_score = best_right_score = 0.5
                best_cx_hoop = best_cy_hoop = best_depth_hoop = best_yaw = None
                best_band_mask = best_left_mask = best_right_mask = best_left_src = best_right_src = None

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.backboard_area: # 1. 일정 넓이 이상
                        rect = cv2.minAreaRect(cnt)  
                        (box_cx, box_cy), (width, height), box_angle = rect # 백보드 정보
                        if width < height:
                            width, height = height, width
                            box_angle += 90.0

                        box = cv2.boxPoints(((box_cx, box_cy), (width, height), box_angle)).astype(np.float32)
                        
                        s = box.sum(axis=1)
                        diff = np.diff(box, axis=1).ravel()
                        tl = box[np.argmin(s)]
                        br = box[np.argmax(s)]
                        tr = box[np.argmin(diff)]
                        bl = box[np.argmax(diff)]
                        src = np.array([tl, tr, br, bl], dtype=np.float32) # 박스 꼭지점 정렬
                        
                        top = max(1, int(round(height * self.band_top_ratio))) # 밴드 두께 - 위
                        side = max(1, int(round(width * self.band_side_ratio))) # 밴드 두께 - 옆
        
                        top_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)
                        left_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)
                        right_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)
                        inner_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)
                        band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8) # 까만 화면 만들고 (0벡터)

                        top_dst = np.array([[0,0],[width,0],[width,top],[0,top]], dtype=np.float32)
                        left_dst = np.array([[0,0],[side,0],[side,height],[0,height]], dtype=np.float32)
                        right_dst = np.array([[width-side,0],[width,0],[width,height],[width-side,height]], dtype=np.float32)
                        inner_dst = np.array([[side,top],[width-side,top],[width-side,height],[side,height]], dtype=np.float32) # 구역 지정
                        
                        dst_rect = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
                        M_inv = cv2.getPerspectiveTransform(dst_rect, src.astype(np.float32)) # 화면 왜곡 보정 값

                        top_src = np.round(cv2.perspectiveTransform(top_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
                        left_src = np.round(cv2.perspectiveTransform(left_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
                        right_src = np.round(cv2.perspectiveTransform(right_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32)
                        inner_src = np.round(cv2.perspectiveTransform(inner_dst.reshape(-1,1,2), M_inv).reshape(-1,2)).astype(np.int32) # 구역 보정

                        cv2.fillPoly(top_mask, [top_src], 255, lineType=cv2.LINE_8)
                        cv2.fillPoly(left_mask, [left_src], 255, lineType=cv2.LINE_8)
                        cv2.fillPoly(right_mask, [right_src], 255, lineType=cv2.LINE_8)
                        cv2.fillPoly(inner_mask, [inner_src], 255, lineType=cv2.LINE_8)
                        cv2.fillPoly(band_mask, [top_src, left_src, right_src], 255, lineType=cv2.LINE_8) 

                        area_top = cv2.countNonZero(top_mask)
                        area_left = cv2.countNonZero(left_mask)
                        area_right = cv2.countNonZero(right_mask)
                        if area_top < 5 or area_left < 5 or area_right < 5: # 구역 넓이 계산
                            continue

                        ratio_top = cv2.countNonZero(cv2.bitwise_and(red_mask, top_mask)) / float(area_top)
                        ratio_left = cv2.countNonZero(cv2.bitwise_and(red_mask, left_mask)) / float(area_left)
                        ratio_right = cv2.countNonZero(cv2.bitwise_and(red_mask, right_mask)) / float(area_right)
                        ratio_band = (ratio_top + ratio_left + ratio_right) / 3 # 구역 내 빨강 비율

                        if ratio_top >= self.red_ratio_min and ratio_left >= self.red_ratio_min and ratio_right >= self.red_ratio_min: # 2. 테두리 빨강 비율 만족
                            area_inner = cv2.countNonZero(inner_mask)
                            if area_inner == 0:
                                continue
                            white_hits = cv2.countNonZero(cv2.bitwise_and(white_mask, inner_mask))
                            ratio_inner = white_hits / float(area_inner)
                            if ratio_inner <= self.white_min_inner: # 3. 내부 하양 비율 만족
                                continue

                            inner_depth_vals = roi_depth[inner_mask.astype(bool)] # 깊이
                            valid = np.isfinite(inner_depth_vals) & (inner_depth_vals > self.depth_min) & (inner_depth_vals < self.depth_max_hoop) # NaN이랑 범위 밖 제외
                            if np.count_nonzero(valid) <= 10: # 3. 깊이 값 만족
                                continue
                            depth_mean = float(np.mean(inner_depth_vals[valid])) * self.depth_scale

                            if best_score < ratio_band: # 그 중 가장 잘 맞는 거
                                best_score, best_top_score, best_left_score, best_right_score = ratio_band, ratio_top, ratio_left, ratio_right
                                best_cnt_hoop = cnt
                                best_cx_hoop, best_cy_hoop = int(round(box_cx + roi_x_start)), int(round(box_cy + roi_y_start))
                                best_depth_hoop = depth_mean
                                best_box = (box + np.array([roi_x_start, roi_y_start], dtype=np.float32)).astype(np.int32)
                                best_band_mask = band_mask
                                best_left_mask = left_mask
                                best_right_mask = right_mask
                                best_left_src = left_src
                                best_right_src = right_src
                t5 = time.time()
                        
                # 백보드 검출은 끝났고 각도 계산 및 정보 갱신
                if best_cnt_hoop is not None:   
                    self.hoop_lost = 0
                    self.rect_color = (0, 255, 0)

                    self.last_score, self.last_top_score, self.last_left_score, self.last_right_score = best_score, best_top_score, best_left_score, best_right_score
                    self.last_cx_hoop, self.last_cy_hoop = best_cx_hoop, best_cy_hoop
                    self.last_z_hoop = best_depth_hoop
                    self.last_box_hoop = best_box
                    self.last_band_mask = best_band_mask

                    left_vals = roi_depth[best_left_mask.astype(bool)] # 왼쪽 깊이
                    right_vals = roi_depth[best_right_mask.astype(bool)] # 오른쪽 깊이

                    left_valid = np.isfinite(left_vals) & (left_vals > self.depth_min) & (left_vals < self.depth_max_hoop)
                    right_valid = np.isfinite(right_vals) & (right_vals > self.depth_min) & (right_vals < self.depth_max_hoop)

                    nL = int(np.count_nonzero(left_valid))
                    nR = int(np.count_nonzero(right_valid)) # 잘 받아오고 있나 개수 검출

                    if nL >= 10 and nR >= 10:
                        depth_left = float(np.mean(left_vals[left_valid])) * self.depth_scale
                        depth_right = float(np.mean(right_vals[right_valid])) * self.depth_scale

                        x_left_px_roi = float(best_left_src[:, 0].mean())
                        x_right_px_roi = float(best_right_src[:, 0].mean())
                        x_left_px_full = x_left_px_roi + roi_x_start
                        x_right_px_full = x_right_px_roi + roi_x_start

                        x_left_m = (x_left_px_full  - self.cx_intr) * depth_left / self.fx
                        x_right_m = (x_right_px_full - self.cx_intr) * depth_right / self.fx # 거리 계산

                        dx = x_right_m - x_left_m
                        dz = depth_left - depth_right  # 오른쪽이 더 가까우면 + >>>> 내가 회전할 각도

                        if abs(dx) > 0.001:  # 1mm
                            best_yaw = math.atan2(dz, dx)
                            is_hoop_valid = True   # 어떤 값 쓸건지 판단용
                            self.last_yaw = best_yaw             

                            self.get_logger().info(f"[HOOP] ZL={depth_left:.3f}m, ZR={depth_right:.3f}m, yaw={best_yaw}, nL={nL}, nR={nR}")
                        else: # 그럴일 없다
                            best_yaw = None
                            is_hoop_valid = False
                            self.get_logger().info(f"[HOOP] Retry : not enough big box")
                    else: # 깊이를 못 받아온 경우 >> 오류거나 반사가 너무 심하거나
                        self.get_logger().info(f"[HOOP] Retry : not enough valid depth")
                        best_yaw = None
                        is_hoop_valid = False

                    self.last_yaw = best_yaw

                    cv2.polylines(frame, [best_box], True, self.rect_color, 2)
                    cv2.circle(frame, (best_cx_hoop, best_cy_hoop), 5, (0, 0, 255), 2)

                    self.last_position_text = f'Dist: {best_depth_hoop:.2f}m | Pos: {best_cx_hoop}, {-best_cy_hoop}, | Acc: {best_score:.2f}, | Ang: {best_yaw}'
                    self.backboard_score_text = f'Top: {best_top_score:.2f} | Left: {best_left_score:.2f} | Right: {best_right_score:.2f}, | Ang: {best_yaw}'

                else:
                    if self.hoop_lost < 3 and self.last_box_hoop is not None:     ## .;;      
                        self.hoop_lost += 1
                        self.rect_color = (255, 0, 0)
                        is_hoop_valid = True

                        best_score, best_top_score, best_left_score, best_right_score = self.last_score, self.last_top_score, self.last_left_score, self.last_right_score
                        best_cx_hoop, best_cy_hoop = self.last_cx_hoop, self.last_cy_hoop
                        best_depth_hoop = self.last_z_hoop 
                        best_box = self.last_box_hoop
                        best_band_mask = self.last_band_mask
                        best_yaw = self.last_yaw

                        cv2.polylines(frame, [best_box], True, self.rect_color, 2)
                        cv2.circle(frame, (best_cx_hoop, best_cy_hoop), 5, (0, 0, 255), 2)

                        self.last_position_text = f'Dist: {best_depth_hoop:.2f}m | Pos: {best_cx_hoop}, {-best_cy_hoop}, | Acc: {best_score:.2f}, | Ang: {best_yaw}'
                        self.backboard_score_text = f'Top: {best_top_score:.2f} | Left: {best_left_score} | Right: {best_right_score:.2f}, | Ang: {best_yaw}'

                    else:
                        self.hoop_lost = 3
                        self.rect_color = (0, 0, 255)
                        is_hoop_valid = False

                        best_cx_hoop = best_cy_hoop = best_depth_hoop = None # cy는 필요없긴 함
                        best_yaw = None

                        self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
                        self.last_cx_hoop = self.last_cy_hoop = None
                        self.last_z_hoop = self.last_yaw = None
                        self.last_box_hoop = None
                        
                        self.last_position_text = f'Miss'
                        self.backboard_score_text = f"Miss"
                        self.last_band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)

                self.frames_left -= 1
                self.hoop_valid_list.append(is_hoop_valid)
                self.hoop_cx_list.append(best_cx_hoop)
                self.hoop_dis_list.append(best_depth_hoop)
                self.yaw_list.append(best_yaw)

                if self.frames_left <= 0:
                    result = Counter(self.hoop_valid_list).most_common(1)[0][0]

                    process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0 # 총 시간
                    angle = 0 # 임시

                    if result == True: # 탐지가 더 많음
                        self.hoop_count += 1
                        cxs = [a for s, a in zip(self.hoop_valid_list, self.hoop_cx_list) if s == result and a is not None]
                        dists = [a for s, a in zip(self.hoop_valid_list, self.hoop_dis_list) if s == result and a is not None]
                        yaws = [a for s, a in zip(self.hoop_valid_list, self.yaw_list) if s == result and a is not None]

                        avg_cx = int(round(np.mean(cxs)))
                        avg_dis = round(np.mean(dists),2)
                        avg_yaw_rad = np.median(yaws)
                        avg_yaw_deg = round(math.degrees(avg_yaw_rad),2)

                        if 0.7 < avg_dis: # 거리가 0.7m 이상 > 일단 골대 쪽으로 직진하기

                            angle = int(round(math.degrees(math.atan((avg_cx - self.cx_intr) / self.fx))))
                            
                            if angle >= 8:
                                res = 14
                            elif angle <= -8:
                                res = 13
                            else:
                                res = 12
                            self.get_logger().info(f"[Hoop] Done: Approaching | x: {avg_cx}, dis: {avg_dis}, "
                                                f"line angle= {angle}, backboard angle= {avg_yaw_deg}")
                            
                        # elif 0.7 <= avg_dis: # 거리가 0.5 ~ 0.8 > 미세 조정
                            # if avg_yaw > 30:
                            #     angle = int(round(avg_yaw))
                            #     res = 7 # 오른쪽으로 꺾고 왼쪽으로 이동
                            # elif avg_yaw < -30:
                            #     angle = int(round(avg_yaw))
                            #     res = 8 # 왼쪽으로 꺾고 오른쪽으로 이동
                            # else:
                            #     angle = 0
                            #     if avg_cx - self.zandi_x > 30:
                            #         res = 7 # 왼쪽 이동
                            #     elif avg_cx - self.zandi_x < -30:
                            #         res = 8 # 오른쪽 이동
                            #     else:
                            #         res = 12
                            # angle = int(round(math.degrees(math.atan(-(avg_cx - self.cx_intr) / self.fx))))
                            
                            # if angle >= 8:
                            #     res = 14
                            # elif angle <= -8:
                            #     res = 13
                            # else:
                            #     res = 12

                            # self.get_logger().info(f"[Hoop] Done: Near by | x: {avg_cx}, dis: {avg_dis}, "
                            #                     f"res= {res}, backboard angle= {avg_yaw_deg} "
                            #                     f"frames= {len(self.hoop_valid_list)}, wall= {process_time*1000:.1f} ms")
                        
                        else: # 거리가 0.7m 이내 > 
                            self.hoop_near_by = True
                            angle = 0
                            beta_rad = math.atan((avg_cx - self.cx_intr) / self.fx) # 백보드 점과의 각도
                            x_ = avg_dis * math.sin(beta_rad) + self.hoop_r * math.sin(avg_yaw_rad)
                            z_ = avg_dis * math.cos(beta_rad) - self.hoop_r * math.cos(avg_yaw_rad) - self.throwing_range # x_, z_는 움직여야할 좌표

                            if math.hypot(x_,z_) < self.goal_range:
                                res = 17
                                self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
                                self.last_cx_hoop = self.last_cy_hoop = self.last_z_hoop = self.last_yaw = None
                                self.last_box_hoop = None
                                self.last_band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)  

                                self.hoop_near_by = False
                                self.cam1_mode = BALL
                                self.backboard_score_text = "Ball Now"

                            elif z_ <= 0: 
                                res = 5 # 뒤로가기

                            elif abs(x_) >= z_:
                                if x_ > 0:
                                    res = 8 # 오른
                                else: 
                                    res = 7 # 왼
                            else:
                                res = 12 # 앞
                                

                            if res == 17:
                                self.get_logger().info(f"[Hoop] Shoot !! | x: {avg_cx}, dis: {avg_dis}, "
                                                f"backboard angle= {avg_yaw_deg}")  

                            else: 
                                self.get_logger().info(f"[Hoop] Done: Almost near by | x: {avg_cx}, dis: {avg_dis}, "
                                                f"backboard angle= {avg_yaw_deg} , beta= {math.degrees(beta_rad)}")

                    else:
                        if self.hoop_near_by: # 근접했는데 놓쳤으면 일단 뒤로가서 다시 봐라
                            res = 5
                            self.get_logger().info(f"[Hoop] Missed,,, go baack one step")
                            angle = 0
                            
                        elif self.hoop_count >= 1: # 근접하기도 전에 놓쳤으면 일단 뒤로가서 확인해봐라
                            res = 5 # 다시 찾는 모션 >> 제자리에서? 한발짝 뒤에서?
                            angle = 0

                            self.get_logger().info(f"[Hoop] Try again,,,")
                            
                            self.hoop_count -= 1
                            self.last_position_text = "No hoop"

                        else: # 아깐 잘못 봤나봐 (두 번이상 놓쳤을 때)
                            self.get_logger().info(f"[Hoop] No hoop detected")
                            
                            self.last_position_text = "No hoop"
                            self.hoop_count = 0

                            res = 5 # 라인 걸으세요 임시
                            angle = 0

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

            cv2.rectangle(frame, (int((roi_x_start + roi_x_end) / 2) - 50, roi_y_start), 
                (int((roi_x_start + roi_x_end) / 2) + 50, roi_y_end), (255, 120, 150), 1)
                
        t6 = time.time()  

        cv2.rectangle(frame, (roi_x_start + 1, roi_y_start + 1), (roi_x_end - 1, roi_y_end - 1), self.rect_color, 1)

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
        cv2.putText(frame, self.last_position_text, (10, roi_y_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        cv2.putText(frame, self.backboard_score_text, (10, roi_y_end + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)

        if self.collecting:
            if self.cam1_mode == BALL:
                cv2.imshow('Basketball Mask', mask) 
            elif self.cam1_mode == HOOP:
                cv2.imshow('Red Mask', red_mask)
                cv2.imshow('White Mask', white_mask)
                if best_band_mask is not None:
                    cv2.imshow('Band Mask', self.last_band_mask)

        cv2.imshow('Detection', frame)
        cv2.waitKey(1)

    def decide_to_pick(self, dx, dy, process_time):
        if abs(dx) <= 30 and abs(dy) <= 20: # 오케이 조준 완료  310 360     315 285
            res = 9 # pick 모션
            self.picked = True
            self.pick_attempt += 1

        # dy가 너뮤 가까우면 뒤로 가게도 하자
        # x합격, y 합격 나누고 둘다 불합격일떄만 크기 비교해서 하기
        elif abs(dx) <= 30 and abs(dy) > 20:
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

        elif abs(dx) > 30 and abs(dy) <= 20:
            if dx > 0: 
                res = 8 #right_half
        
            elif dx < 0:
                res = 7 #left_half
        
        elif abs(dx) > 30 and abs(dy) > 20:
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
        roi_color = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]


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
                self.ball_lost = 0

                self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                
                self.line_start_time = time.time()
                
                self.draw_color = (0, 255, 0)
                self.rect_color = (0, 255, 0)

                self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames in CAM{self.cam_mode}')
        
        if self.collecting:
            
            angle = 0
            self.frame_idx += 1
            self.get_logger().info(f"{self.frame_idx}")
            
            # HSV 색 조절
            self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            raw_mask = cv2.inRange(self.hsv, self.lower_hsv_ball, self.upper_hsv_ball) # 주황색 범위 색만
            
            # 모폴로지 연산
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어

            best_cnt_ball = None
            best_ratio_ball = 0.3
            best_cx_ball = best_cy_ball = 0

            for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
                area = cv2.contourArea(cnt) # 1. 면적 > 1000
                if area > 1000:
                    (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                    circle_area = circle_r * circle_r * math.pi
                    ratio = abs((area / circle_area) - 1)
                    # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈1
                    if ratio < best_ratio_ball:
                        best_ratio_ball = ratio
                        best_cnt_ball = cnt
                        best_cx_ball = int(x + roi_x_start)
                        best_cy_ball = int(y + roi_y_start)
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

                self.get_logger().info(f"Found! {best_cx_ball}, {best_cy_ball}")

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

                    self.get_logger().info(f"Lost! {best_cx_ball}, {best_cy_ball}")

                    cv2.circle(frame, [best_cx_ball, best_cy_ball], best_radius, self.draw_color, 2)
                else:
                    self.ball_lost = 3
                    best_cx_ball = best_cy_ball = None

                    self.last_cx_ball = None
                    self.last_cy_ball = None
                    self.last_radius = None

                    self.rect_color = (0, 0, 255)
                    is_ball_valid = False
                    self.get_logger().info(f"Miss!")
            
            self.frames_left -= 1
            self.ball_valid_list.append(is_ball_valid)
            self.ball_cx_list.append(best_cx_ball)
            self.ball_cy_list.append(best_cy_ball)

            if self.frames_left <= 0:
                result = Counter(self.ball_valid_list).most_common(1)[0][0]

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

                    if self.picked: # 줍기 모션을 했다면 >> 나 방금 주웠는데 눈 앞에 공이 또 있네?
                        if self.pick_attempt >= 3: # 3번 넘게 실패
                            res = 22
                            self.get_logger().info(f"[Ball] Failed to pick,,, I'll give up | Pos : {dx}, {-dy} | attempt= {self.pick_attempt}")

                            self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                            self.pick_attempt = 0
                            self.backboard_score_text = "Hoop Now"

                            self.picked = False
                            self.ball_saw_once = False
                            self.ball_never_seen = True
                            self.cam_mode = CAM1
                            self.cam1_mode = BALL # 공 못 주웠으니, 골대 찾을 필요가 없다 

                        else: # 시도는 3번만
                            res = self.decide_to_pick(dx, dy, process_time)
                            self.picked = True
                            self.get_logger().info(f"[Ball] Pick one more time! | Pos : {dx}, {-dy}, attempt {self.pick_attempt}")

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

                        self.last_position_text = f"[Ball] Position: {avg_cx}, {avg_cy}"
                        res = self.decide_to_pick(dx, dy, process_time)
                        self.get_logger().info(f"[Ball] Pick! | Pos : {dx}, {-dy}, attempt {self.pick_attempt}")

                else: # 탐지 실패
                    self.get_logger().info(f"picked = {self.picked}")
                    if self.picked: # 방금 줍는 모션을 했는데 내 눈앞에 공이 없다 = 잘 주웠다 (아니면 흘렸거나)
                        self.get_logger().info(f"[Ball] I made it! | , attempt {self.pick_attempt}")
                        res = 10 # 재그립 모션

                        self.last_cx_ball = self.last_cy_ball = self.last_radius = None
                        self.backboard_score_text = "Hoop Now"
                        self.last_position_text = f""

                        self.cam2_miss_count = 0
                        self.pick_attempt = 0
                        self.picked = False
                        self.ball_saw_once = False
                        self.ball_never_seen = True

                        self.cam_mode = CAM1
                        self.cam1_mode = HOOP # 골대 찾으러 가자

                    else: # 공을 탐지도 못 했고, 줍지도 않았다
                        if self.ball_saw_once: # 근데 그 전에 공을 발견한 적이 있긴 함
                            self.cam2_miss_count += 1

                            if self.cam2_miss_count >= 5: # 한 번 봐놓고 공을 5번 연속이나 못 보면 뭐할까?  >>>  뭔가 찾는 모션
                                self.get_logger().info(f"[Ball] I totally missed,,,")
                                res = 5
                                self.last_position_text = f""
                            
                            else: # 한번만 다시 봐봐
                                self.get_logger().info(f"[Ball] Retry,,, | miss= {self.cam2_miss_count}")
                                res = 5
                                self.last_position_text = "Miss"
                                # 오른쪽 움직일 때 회전하느라공 뒤로 사라지는 거 생각하기
                        
                        else: # 공을 탐지도 못 했고, 줍지도 않았고 공을 본 적도 없음 = 방금 막 HOOP모드에서 바뀜 >> 직진만 하면서 찾아보자
                            self.get_logger().info(f"[Ball] Finding,,,")
                            res = 12
                            self.last_position_text = f"finding"
        
                msg_out = BallResult()
                msg_out.res = res
                msg_out.angle = angle
                self.ball_result_pub.publish(msg_out)
                self.get_logger().info(f"res = {res}")
                self.get_logger().info(f"frames= {len(self.ball_valid_list)}, wall= {process_time*1000:.1f} ms")
                # 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0

                self.last_position_text = ""

        else:
            self.last_position_text = "In move"

        cv2.rectangle(frame, (roi_x_start+1, roi_y_start+1), (roi_x_end-1, roi_y_end-1), self.rect_color, 1)
        # cv2.circle(frame, (self.pick_x, self.pick_y), self.pick_rad, (111,255,111), 2)
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
        cv2.putText(frame, self.last_position_text, (10, roi_y_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        cv2.putText(frame, self.backboard_score_text, (10, roi_y_end + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)

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