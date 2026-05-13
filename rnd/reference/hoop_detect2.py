#!/usr/bin/env python3

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rcl_interfaces.msg import SetParametersResult
from collections import deque, Counter
from robot_msgs.msg import HoopResult, MotionEnd
from std_msgs.msg import Float32MultiArray


import rclpy
import cv2
import numpy as np
import time
import math

camera_width = 640
camera_height = 480

roi_x_start = int(camera_width * 0 // 5)
roi_x_end = int(camera_width * 5 // 5)
roi_y_start = int(camera_height * 1 // 12)
roi_y_end = int(camera_height * 11 // 12)

class HoopDetectorNode(Node):
    def __init__(self):
        super().__init__('hoop_detector')

        self.hsv = None

        # zandi
        self.zandi_x = int((roi_x_start + roi_x_end) / 2)
        self.zandi_y = int(camera_height - 100)

        # pick
        self.pick_x = self.zandi_x + 15
        self.pick_y = self.zandi_y - 80 
        self.pick_rad = 20

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"
        self.last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력

        self.backboard_score_text = 'Miss' # H

        self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
        self.last_cx_hoop = self.last_cy_hoop = self.last_z_hoop = self.last_yaw = None
        self.last_box_hoop = None
        self.hoop_lost = 0 # H

        # 변수
        self.draw_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.fx, self.fy = 607.0, 606.0   # ros2 topic echo /camera/color/camera_info - 카메라 고유값
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.last_band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype = np.uint8)

        self.lower_hsv_ball = np.array([8, 60, 60])
        self.upper_hsv_ball = np.array([60, 255, 255]) # 주황색 기준으로
        
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
        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)

        self.cam1_ball_count = 0
        self.cam2_miss_count = 0
        self.hoop_count = 0

        self.hoop_valid_list = [] # H
        self.hoop_cx_list = []
        self.hoop_dis_list = []
        self.yaw_list = []

        self.bridge = CvBridge()   
                              
        self.cam1_color_sub = Subscriber(self, Image, '/camera/color/image_raw') # CAM1
        self.cam1_depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw') 
        self.cam1_sync = ApproximateTimeSynchronizer([self.cam1_color_sub, self.cam1_depth_sub], queue_size=5, slop=0.1)
        self.cam1_sync.registerCallback(self.image_callback)
        
        self.motion_end_sub = self.create_subscription(  # 모션엔드
            MotionEnd,                   
            '/motion_end',                   
            self.motion_callback, 10) 
        
        self.hoop_result_pub = self.create_publisher(HoopResult, '/hoop_result', 10)

        # 파라미터 선언 H
        self.declare_parameter('red_h1_low', 0) # 빨강
        self.declare_parameter('red_h1_high', 10)
        self.declare_parameter('red_h2_low', 160)
        self.declare_parameter('red_h2_high', 180)
        self.declare_parameter('red_s_low', 80)
        self.declare_parameter('red_v_low', 60)
        
        self.declare_parameter('white_s_high', 155) # 하양 70. 190
        self.declare_parameter('white_v_low', 90)

        self.declare_parameter('band_top_ratio', 0.15)   # 백보드 h x 0.15
        self.declare_parameter('band_side_ratio', 0.10)   # w x 0.10

        self.declare_parameter('red_ratio_min', 0.55)     # 백보드 영역
        self.declare_parameter('white_min_inner', 0.50)  
        self.declare_parameter('backboard_area', 1500)

        # SIM
        self.declare_parameter('sim_mode', False)      # True면 카메라 대신 /hoop_sim/data 사용
        self.declare_parameter('sim_auto_arm', True)   # 자동으로 window 시작(armed 토글)
        self.declare_parameter('sim_arm_period', 0.40) # 초, 윈도우 시작 간격

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

        # SIM
        self.sim_mode = bool(self.get_parameter('sim_mode').value)
        self.sim_auto_arm = bool(self.get_parameter('sim_auto_arm').value)
        self.sim_arm_period = float(self.get_parameter('sim_arm_period').value)

        self.add_on_set_parameters_callback(self.param_callback)

        if self.sim_mode:
            # 1) 카메라 동기화 대신, 의사 데이터 구독
            self.sim_sub = self.create_subscription(
                Float32MultiArray, '/hoop_sim/data', self.sim_cb, 10
            )

            # 2) 자동 윈도우 트리거 (모션엔드 없이도 주기적으로 collecting 시작)
            if self.sim_auto_arm:
                self.arm_timer = self.create_timer(self.sim_arm_period, self._sim_arm_tick)

        else:
            # 기존 카메라 동기화 로직 그대로 유지
            self.cam1_color_sub = Subscriber(self, Image, '/camera/color/image_raw') # CAM1
            self.cam1_depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw') 
            self.cam1_sync = ApproximateTimeSynchronizer([self.cam1_color_sub, self.cam1_depth_sub], queue_size=5, slop=0.1)
            self.cam1_sync.registerCallback(self.image_callback)

        # 클릭
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ball Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Detection', self.on_click)
        cv2.setMouseCallback('Ball Detection', self.on_click)

    def param_callback(self, params): # 파라미터 변경 적용
        for param in params:
            if param.name == "red_h1_low":
                if param.value >= 0 and param.value <= self.red_h1_high:
                    self.red_h1_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_h1_high":
                if param.value >= self.red_h1_low and param.value <= 179:
                    self.red_h1_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_h2_low":
                if param.value >= 0 and param.value <= self.red_h2_high:
                    self.red_h2_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_h2_high":
                if param.value >= self.red_h2_low and param.value <= 179:
                    self.red_h2_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_s_min":
                if param.value >= 0 and param.value <= 255:
                    self.red_s_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_v_min":
                if param.value >= 0 and param.value <= 255:
                    self.red_v_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "white_s_max":
                if param.value >= 0 and param.value <= 255:
                    self.white_s_max = param.value
                    self.get_logger().info(f"Changed")
                else:
                    return SetParametersResult(successful=False)
            if param.name == "white_v_min":
                if param.value >= 0 and param.value <= 255:
                    self.white_v_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "band_top_ratio":
                if param.value > 0 and param.value < 1:
                    self.band_top_ratio = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "band_side_ratio":
                if param.value > 0 and param.value < 1:
                    self.band_side_ratio = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "red_ratio_min":
                if param.value > 0 and param.value < 1:
                    self.red_ratio_min = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "white_min_inner":
                if param.value > 0 and param.value < 1:
                    self.white_min_inner = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "backboard_area":
                if param.value > 0 and param.value <= 5000:
                    self.backboard_area = param.value
                else:
                    return SetParametersResult(successful=False)
            
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
        self.get_logger().info(f"[Pos] x={x - self.zandi_x}, y={-(y - self.zandi_y)} | HSV=({H},{S},{V})")

    def motion_callback(self, msg: MotionEnd): # 모션 끝 같이 받아오기 (중복 방지)
        if bool(msg.motion_end_detect):
            self.armed = True
            self.get_logger().info("Subscribed /motion_end !!!!!!!!!!!!!!!!!!!!!!!!")

    def _sim_arm_tick(self):
        # collecting 중이 아닐 때만 새 윈도우 시작
        if not self.collecting:
            self.armed = True
            self.get_logger().info("[SIM] auto armed")

    def sim_cb(self, msg: Float32MultiArray):
        # msg.data = [valid(0/1), cx(px), dist(m), yaw_deg]
        try:
            valid = bool(int(msg.data[0]))
            cx    = int(round(float(msg.data[1])))
            dist  = float(msg.data[2])
            yaw   = float(msg.data[3])   # deg
        except Exception as e:
            self.get_logger().warn(f"[SIM] bad message: {e}")
            return

        # 윈도우 시작(armed → collecting) 처리
        if not self.collecting:
            if self.armed:
                self.collecting = True
                self.armed = False
                self.frames_left = self.collecting_frames
                self.window_id += 1

                # 윈도우 초기화(너 코드와 동일)
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

                self.get_logger().info(f'[SIM] Start Window {self.window_id} | {self.collecting_frames} frames')

        # 수집 중이면 push
        if self.collecting:
            self.frame_idx += 1
            self.frames_left -= 1
            self.hoop_valid_list.append(valid)
            self.hoop_cx_list.append(cx if valid else None)
            self.hoop_dis_list.append(dist if valid else None)
            self.yaw_list.append(yaw if valid else None)

            # 윈도우 종료 시, 네 “결정 로직” 그대로 실행
            if self.frames_left <= 0:
                result = Counter(self.hoop_valid_list).most_common(1)[0][0]
                process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                angle = 0
                if result is True:
                    self.hoop_count += 1
                    cxs   = [a for s,a in zip(self.hoop_valid_list, self.hoop_cx_list) if s and a is not None]
                    dists = [a for s,a in zip(self.hoop_valid_list, self.hoop_dis_list) if s and a is not None]
                    yaws  = [a for s,a in zip(self.hoop_valid_list, self.yaw_list) if s and a is not None]

                    # 빈 리스트 가드
                    if not cxs or not dists or not yaws:
                        res = 99
                        angle = 0
                        self.get_logger().info(f"[SIM Hoop] Not enough valid samples | frames= {len(self.hoop_valid_list)}, wall= {process_time*1000:.1f} ms")
                    else:
                        avg_cx = int(round(np.mean(cxs)))
                        avg_dis = round(float(np.mean(dists)), 2)
                        avg_yaw = round(float(np.median(yaws)), 2)  # deg

                        if 0.8 < avg_dis:
                            # 원거리: 중심각으로 13/14/12
                            angle = int(round(math.degrees(math.atan(-(avg_cx - self.cx_intr) / self.fx))))
                            if angle >= 8:   res = 14
                            elif angle <= -8:res = 13
                            else:            res = 12
                            self.get_logger().info(f"[SIM Hoop] Approaching | x:{avg_cx}, dis:{avg_dis}, res={res}, line angle={angle}, backboard angle={avg_yaw} frames={len(self.hoop_valid_list)}, wall={process_time*1000:.1f} ms")

                        elif 0.5 <= avg_dis:
                            # 중거리: 너 코드 버전(백보드 yaw 우선 → 7/8, 아니면 cx 오프셋 → 7/8, 아니면 12)
                            if avg_yaw > 30:
                                angle = int(round(avg_yaw))
                                res = 7
                            elif avg_yaw < -30:
                                angle = int(round(avg_yaw))
                                res = 8
                            else:
                                angle = 0
                                if avg_cx - self.zandi_x > 30:   res = 7
                                elif avg_cx - self.zandi_x < -30:res = 8
                                else:                             res = 12
                            self.get_logger().info(f"[SIM Hoop] Near by | x:{avg_cx}, dis:{avg_dis}, res={res}, backboard angle={avg_yaw} frames={len(self.hoop_valid_list)}, wall={process_time*1000:.1f} ms")

                        else:
                            # 근거리: 너 코드 버전(82/83/74/75/77)
                            if avg_yaw > 30:
                                angle = int(round(avg_yaw)); res = 82
                            elif avg_yaw < -30:
                                angle = int(round(avg_yaw)); res = 83
                            else:
                                angle = 0
                                if   avg_cx - self.zandi_x >  50: res = 74
                                elif avg_cx - self.zandi_x < -50: res = 75
                                else:                              res = 77

                            if res == 77:
                                self.get_logger().info(f"[SIM Hoop] Shoot !! | x:{avg_cx}, dis:{avg_dis}, res={res}, backboard angle={avg_yaw} frames={len(self.hoop_valid_list)}, wall={process_time*1000:.1f} ms")
                                # 상태 초기화(실기와 동일)
                                self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
                                self.last_cx_hoop = self.last_cy_hoop = self.last_z_hoop = self.last_yaw = None
                                self.last_box_hoop = None
                                self.last_band_mask = np.zeros((roi_y_end - roi_y_start), dtype=np.uint8)
                                self.backboard_score_text = "Ball Now"
                            else:
                                self.get_logger().info(f"[SIM Hoop] Almost near | x:{avg_cx}, dis:{avg_dis}, res={res}, backboard angle={avg_yaw} frames={len(self.hoop_valid_list)}, wall={process_time*1000:.1f} ms")

                else:
                    if self.hoop_count >= 1:
                        res = 55; angle = 0
                        self.get_logger().info(f"[SIM Hoop] Missed,,, | frames={len(self.hoop_valid_list)}, wall={process_time*1000:.1f} ms")
                        self.hoop_count = 0
                        self.last_position_text = "No hoop"
                    else:
                        res = 99; angle = 0
                        self.get_logger().info(f"[SIM Hoop] No hoop detected | frames={len(self.hoop_valid_list)}, wall={process_time*1000:.1f} ms")
                        self.last_position_text = "No hoop"
                        self.hoop_count = 0

                # 퍼블리시 (실기와 동일)
                msg_out = HoopResult()
                msg_out.res = res
                msg_out.angle = abs(int(angle)) if isinstance(angle, (int,float)) else 0
                self.hoop_result_pub.publish(msg_out)

                # 윈도우 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0


    def image_callback(self, color_msg: Image, depth_msg: Image):
        start_time = time.time()

        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)  # mm

        # ROI
        roi_color = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_depth = depth[roi_y_start:roi_y_end, roi_x_start:roi_x_end]  # mm

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
                    depth_right = float(np.mean(right_vals[right_valid]))* self.depth_scale

                    x_left_px_roi = float(best_left_src[:, 0].mean())
                    x_right_px_roi = float(best_right_src[:, 0].mean())
                    x_left_px_full = x_left_px_roi + roi_x_start
                    x_right_px_full = x_right_px_roi + roi_x_start

                    x_left_m = (x_left_px_full  - self.cx_intr) * depth_left / self.fx
                    x_right_m = (x_right_px_full - self.cx_intr) * depth_right / self.fx # 거리 계산

                    dx = x_right_m - x_left_m
                    dz = depth_left - depth_right  # 오른쪽이 더 가까우면 + >>>> 내가 회전할 각도

                    if abs(dx) > 0.001:  # 1mm
                        best_yaw = round(math.degrees(math.atan2(dz, dx)), 2)
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

                if result == True: # 탐지가 더 많음
                    self.hoop_count += 1
                    cxs = [a for s, a in zip(self.hoop_valid_list, self.hoop_cx_list) if s == result and a is not None]
                    dists = [a for s, a in zip(self.hoop_valid_list, self.hoop_dis_list) if s == result and a is not None]
                    yaws = [a for s, a in zip(self.hoop_valid_list, self.yaw_list) if s == result and a is not None]

                    avg_cx = int(round(np.mean(cxs)))
                    avg_dis = round(np.mean(dists),2)
                    avg_yaw = round(np.median(yaws),2)

                    if 0.8 < avg_dis: # 거리가 0.7m 이상 > 일단 골대 쪽으로 직진하기

                        angle = int(round(math.degrees(math.atan(-(avg_cx - self.cx_intr) / self.fx))))
                        
                        if angle >= 8:
                            res = 14
                        elif angle <= -8:
                            res = 13
                        else:
                            res = 12
                        self.get_logger().info(f"[Hoop] Done: Approaching | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, line angle= {angle}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.hoop_valid_list)}, wall= {process_time*1000:.1f} ms")
                        
                    elif 0.5 <= avg_dis: # 거리가 0.5 ~ 0.8 > 미세 조정
                        if avg_yaw > 30:
                            angle = int(round(avg_yaw))
                            res = 7
                        elif avg_yaw < -30:
                            angle = int(round(avg_yaw))
                            res = 8
                        else:
                            angle = 0
                            if avg_cx - self.zandi_x > 30:
                                res = 7
                            elif avg_cx - self.zandi_x < -30:
                                res = 8
                            else:
                                res = 12

                        self.get_logger().info(f"[Hoop] Done: Near by | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.hoop_valid_list)}, wall= {process_time*1000:.1f} ms")
                    
                    else: # 거리가 0.5m 이내 > 
                        if avg_yaw > 30:
                            angle = int(round(avg_yaw))
                            res = 82 # 오른쪽으로 꺾고 왼쪽으로 이동
                        elif avg_yaw < -30:
                            angle = int(round(avg_yaw))
                            res = 83 # 왼쪽으로 꺾고 오른쪽으로 이동
                        else:
                            angle = 0
                            if avg_cx - self.zandi_x > 50:
                                res = 74 # 왼쪽 반 이동
                            elif avg_cx - self.zandi_x < -50:
                                res = 75 # 오른쪽 반 이동
                            else:
                                res = 77 # 던져 던져

                        if res == 77:
                            self.get_logger().info(f"[Hoop] Shoot !! | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.hoop_valid_list)}, wall= {process_time*1000:.1f} ms")  

                            self.last_score = self.last_top_score = self.last_left_score = self.last_right_score = None
                            self.last_cx_hoop = self.last_cy_hoop = self.last_z_hoop = self.last_yaw = None
                            self.last_box_hoop = None
                            self.last_band_mask = np.zeros((roi_y_end - roi_y_start, roi_x_end - roi_x_start), dtype=np.uint8)  

                            self.backboard_score_text = "Ball Now"

                        else: 
                            self.get_logger().info(f"[Hoop] Done: Almost near by | x: {avg_cx}, dis: {avg_dis}, "
                                            f"res= {res}, backboard angle= {avg_yaw} "
                                            f"frames= {len(self.hoop_valid_list)}, wall= {process_time*1000:.1f} ms")

                else:
                    if self.hoop_count >= 1: # 그 전까지 공을 보고 있었다면
                        res = 55 # 다시 찾는 모션 >> 제자리에서? 한발짝 뒤에서?
                        angle = 0

                        self.get_logger().info(f"[Hoop] Missed,,, | frames= {len(self.hoop_valid_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")
                        
                        self.hoop_count = 0
                        self.last_position_text = "No hoop"

                    else: 
                        self.get_logger().info(f"[Hoop] No hoop detected | frames= {len(self.hoop_valid_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")
                        
                        self.last_position_text = "No hoop"
                        self.hoop_count = 0

                        res = 99 # 라인 걸으세요
                        angle = 0

                # 퍼블리시
                msg_out = HoopResult()
                msg_out.res = res
                msg_out.angle = abs(angle)
                self.hoop_result_pub.publish(msg_out)

                # 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0

        cv2.rectangle(frame, (int((roi_x_start + roi_x_end) / 2) - 50, roi_y_start), 
            (int((roi_x_start + roi_x_end) / 2) + 50, roi_y_end), (255, 120, 150), 1)
                
        # 출력
        cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), self.rect_color, 1)
        cv2.rectangle(frame, (int((roi_x_start + roi_x_end) / 2) - 50, roi_y_start), 
                        (int((roi_x_start + roi_x_end) / 2) + 50, roi_y_end), (255, 120, 150), 1)
        cv2.circle(frame, (self.zandi_x, self.zandi_y), 5, (255, 255, 255), -1)
                
        t6 = time.time()        

        # 시간
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time  += elapsed
        now = time.time()

        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / max(1, self.frame_count)
            fps = self.frame_count / max(1e-6, (now - self.last_report_time))
            self.last_avg_text = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        cv2.putText(frame, self.backboard_score_text, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)
        
        # self.get_logger().info(f"ROI = {(t1-start_time)*1000:.3f}, Conv = {(t2-t1)*1000:.3f}, InRange = {(t3 - t2)*1000:.3f},"
        #                        f"Mophology = {(t4 - t3)*1000:.3f}, "
        #                        f"Contour = {(t5 - t4)*1000:.3f}, Decision = {(t6 - t5)*1000:.3f}, Show = {(now - t6)*1000:.3f}")

        if self.collecting:
            cv2.imshow('Red Mask', red_mask)
            cv2.imshow('White Mask', white_mask)
        cv2.imshow('Band Mask', self.last_band_mask)
        cv2.imshow('Hoop Detection', frame)
        cv2.waitKey(1)

def main():

    rclpy.init()
    node = HoopDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()