#!/usr/bin/env python3
import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque, Counter
from robot_msgs.msg import LinePoint, LinePointsArray, BallResult, MotionEnd # type: ignore
from rcl_interfaces.msg import SetParametersResult
from message_filters import Subscriber, ApproximateTimeSynchronizer # type: ignore  동기화

CAM1, CAM2 = 1, 2
pick_rad = 20

class LineListenerNode(Node): #################################################################### 판단 프레임 수 바꿀 때 yolo_cpp 도 고려해라~~~
    def __init__(self):
        super().__init__('line_subscriber')
        
        # 공용 변수
        self.image_width = 640
        self.image_height = 480

        self.roi_x_start = int(self.image_width * 0 // 5)  # 초록 박스 관심 구역
        self.roi_x_end   = int(self.image_width * 5 // 5)
        self.roi_y_start = int(self.image_height * 1 // 12)
        self.roi_y_end   = int(self.image_height * 11 // 12)

        # zandi
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # pick
        self.pick_x = self.zandi_x - 100
        self.pick_y = self.zandi_y  - 100

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"
        self.last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력

        # 추적
        self.last_cx_ball = None
        self.last_cy_ball = None
        self.last_z = None
        self.lost = 0

        # 변수
        self.ball_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.fx, self.fy = 607.0, 606.0   # ros2 topic echo /camera/color/camera_info - 카메라 고유값
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.lower_hsv = np.array([8, 60, 60])
        self.upper_hsv = np.array([60, 255, 255]) # 주황색 기준으로
        self.depth_thresh = 1500.0  # mm 기준 마스크 거리
        self.depth_scale = 0.001    # m 변환용

        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)
        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        self.collecting_frames = 15
        
        self.collecting = False     # 수집 중 여부

        self.armed = False               # motion_end 방어~!
        self.window_id = 0
        self.frame_idx = 0       

        self.cam1_ball_count = 0
        self.cam2_miss_count = 0

        self.last_angle = 0 
        self.last_agv_cx = 0  
        self.last_agv_cy = 0
        
        self.dis_list = []
        self.lost_list = []
        self.cx_list = []
        self.cy_list = []

        self.bridge = CvBridge() 

        self.motion_end_sub = self.create_subscription(  # 모션 끝나면 받아올 T/F
            MotionEnd,                   
            '/motion_end',                   
            self.motion_callback, 10)   
                              
        self.cam1_color_sub = Subscriber(self, Image, '/cam1/color/image_raw') # 칼라
        self.cam1_depth_sub = Subscriber(self, Image, '/cam1/aligned_depth_to_color/image_raw') # 깊이
        self.cam1_sync = ApproximateTimeSynchronizer([self.cam1_color_sub, self.cam1_depth_sub], queue_size=5, slop=0.1)
        self.cam1_sync.registerCallback(self.cam1_image_callback)

        self.cam2_color_sub = self.create_subscription(  # 이미지 토픽
            Image,
            '/cam2/color/image_raw',  #  640x480 / 15fps
            self.cam2_image_callback, 10)
        
        # 파라미터 선언
        self.declare_parameter("cam_mode", CAM1)

        self.declare_parameter("h_low", 8) 
        self.declare_parameter("h_high", 60)  
        self.declare_parameter("s_low", 60)  
        self.declare_parameter("s_high", 255)  
        self.declare_parameter("v_low", 0)  
        self.declare_parameter("v_high", 255)

        # 파라미터 적용
        self.cam_mode = self.get_parameter("cam_mode").value

        self.h_low = self.get_parameter("h_low").value
        self.h_high = self.get_parameter("h_high").value
        self.s_low = self.get_parameter("s_low").value
        self.s_high = self.get_parameter("s_high").value
        self.v_low = self.get_parameter("v_low").value
        self.v_high = self.get_parameter("v_high").value

        self.lower_hsv = np.array([self.h_low, self.s_low, self.v_low ], dtype=np.uint8) # 색공간 미리 선언
        self.upper_hsv = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
        self.hsv = None 

        # 퍼블리셔
        self.ball_result_pub = self.create_publisher(BallResult, '/ball_result', 10)

        # 클릭
        cv2.namedWindow('Hoop Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Hoop Detection', self.on_click)
        
    def on_click(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        if not (self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end):
            return

        rx = x - self.roi_x_start
        ry = y - self.roi_y_start
        k = 1  # (3,3)

        y0, y1 = max(0, ry - k), min(self.roi_y_end - self.roi_y_start, ry + k + 1)
        x0, x1 = max(0, rx - k), min(self.roi_x_end - self.roi_x_start, rx + k +1)
        patch = self.hsv[y0:y1, x0:x1].reshape(-1,3)

        H, S, V = np.mean(patch, axis=0).astype(int)
        self.get_logger().info(f"[Pos] x={x - self.zandi_x}, y={-(y - self.zandi_y)} | HSV=({H},{S},{V})")

    def motion_callback(self, msg: MotionEnd): # 모션 끝 같이 받아오기 (중복 방지)
        if bool(msg.end):
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

        if not self.collecting:
            if self.armed:
                self.collecting = True
                self.armed = False
                self.frames_left = self.collecting_frames
                self.window_id += 1
                
                # 초기화
                self.lost_list.clear()
                self.cx_list.clear()
                self.cy_list.clear()
                self.dis_list.clear()
                self.frame_idx = 0

                self.lost = 0

                self.cam2_miss_count = 0

                self.line_start_time = time.time()

                self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames in CAM{self.cam_mode}')

        if self.collecting:
            self.frame_idx += 1
            self.get_logger().info(f"step {self.frame_idx}")

            # HSV 색 조절
            self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            raw_mask = cv2.inRange(self.hsv, self.lower_hsv, self.upper_hsv) # 주황색 범위 색만
            raw_mask[roi_depth >= self.depth_thresh] = 0  
            raw_mask[roi_depth <= 0.1] = 0 

            # 모폴로지 연산
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

            # 컨투어
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어
            best_cnt = None
            best_ratio = 1
            x_best = y_best = None
            for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
                area = cv2.contourArea(cnt) # 1. 면적 > 1000
                if area > 500:
                    (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                    circle_area = circle_r * circle_r * math.pi
                    ratio = abs((area / circle_area) - 1)
                    # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈
                    if ratio < best_ratio and ratio < 0.3:
                        best_ratio = ratio
                        best_cnt = cnt
                        x_best = int(x)
                        y_best = int(y)

            # 검출 결과 처리: 이전 위치 유지 로직
            if best_cnt is not None:
                # 원 탐지를 했다면
                self.lost = 0
                cx_ball = x_best + self.roi_x_start
                cy_ball = y_best + self.roi_y_start  # 전체화면에서 중심 좌표
                
                x1 = max(x_best - 1, self.roi_x_start)
                x2 = min(x_best + 2, self.roi_x_end)
                y1 = max(y_best - 1, self.roi_y_start)
                y2 = min(y_best + 2, self.roi_y_end)

                roi_patch = roi_depth[y1:y2, x1:x2]
                z = float(roi_patch.mean()) * self.depth_scale # 3x3 거리
                
                # 이전 위치 업데이트
                self.last_cx_ball = cx_ball
                self.last_cy_ball = cy_ball
                self.last_z = z

                self.rect_color = (0, 255, 0)
                self.ball_color = (255, 0, 0)

            elif self.lost < 3 and self.last_cx_ball is not None:
                # 최근 10프레임 내에는 이전 위치 유지
                self.lost += 1
                cx_ball = self.last_cx_ball
                cy_ball = self.last_cy_ball
                z = self.last_z
                self.ball_color = (0, 255, 255)
                self.rect_color = (0, 255, 255)
            else:
                # Miss 상태
                self.lost = 3
                # 표시할 위치 없음
                cx_ball = cy_ball = z = -1
                self.rect_color = (0, 0, 255)
            
            self.get_logger().info(f"{cx_ball}, {cy_ball}, dist = {z}")

            self.frames_left -= 1
            self.lost_list.append(self.lost - 3 == 0)
            self.cx_list.append(cx_ball)
            self.cy_list.append(cy_ball)
            self.dis_list.append(z)

            if self.frames_left <= 0:
                cnt = Counter(self.lost_list)
                result = max(cnt.items(), key=lambda kv: kv[1])[0]

                process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                if result != True:
                    cxs = [a for s, a in zip(self.lost_list, self.cx_list) if s == result]
                    cys = [a for s, a in zip(self.lost_list, self.cy_list) if s == result]
                    dists = [a for s, a in zip(self.lost_list, self.dis_list) if s == result]
                    avg_cx = int(round(np.mean(cxs)))
                    avg_cy = int(round(np.mean(cys)))
                    avg_dis = np.mean(dists)

                    angle = int(round(math.degrees(math.atan2(avg_cx - self.zandi_x, -avg_cy + self.zandi_y))))

                    if angle > 90:
                        angle -= 180
                    
                    if angle >= 8:
                        res = 14
                    elif angle <= -8:
                        res = 13
                    else:
                        res = 12

                    self.get_logger().info(f"[Ball] Done: CAM1 found ball | {avg_cx}, {avg_cy}, dis: {avg_dis:.2f}, "
                                        f"res= {res}, angle= {angle} "
                                        f"frames= {len(self.lost_list)}, wall= {process_time*1000:.1f} ms")
                    self.last_angle = angle
                    self.last_agv_cx = avg_cx
                    self.last_agv_cy = avg_cy
                    self.last_position_text = f"[Ball] Position: {avg_cx}, {avg_cy}"
                    
                    # 퍼블리시
                    msg_out = BallResult()
                    msg_out.res = res
                    #msg_out.angle = abs(angle)
                    self.ball_result_pub.publish(msg_out)

                    self.cam1_ball_count += 1

                else:
                    if self.last_agv_cy >= 300 and self.last_z <= 1 and self.cam1_ball_count >= 1: # 그 전까지 공을 보고 있었다면
                        res = 12
                        angle = 0
                        self.cam_mode = CAM2 # 2번 캠으로 ㄱㄱ

                        self.get_logger().info(f"[Ball] CAM1 Missed, CAM2 will find,,, | frames= {len(self.lost_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")
                        
                        self.last_position_text ="Recieving,,,"
                        self.cam1_ball_count = 0

                        # 퍼블리시
                        msg_out = BallResult()
                        msg_out.res = res
                        #msg_out.angle = abs(angle)
                        self.ball_result_pub.publish(msg_out)

                    elif self.cam1_ball_count == 1:
                        self.get_logger().info(f"[Ball] Wrong thing | frames= {len(self.lost_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")

                        self.last_position_text = "Back to line,,,"
                        self.cam1_ball_count = 0
                    
                    else: 
                        self.get_logger().info(f"[Ball] No ball detected | frames= {len(self.lost_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")
                        
                        self.last_position_text = "No ball"
                        self.cam1_ball_count = 0

                # 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0

                self.last_position_text = ""

            # ROI랑 속도 표시
            cv2.rectangle(frame, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), self.rect_color, 1)
            cv2.circle(frame, (self.zandi_x, self.zandi_y), 3, (255, 255, 255), -1)

            # 후처리 된 마스킹 영상
            # final_mask = np.zeros_like(mask)
            # if best_cnt is not None:
            #     cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
        else:
            self.last_position_text = "In move"

        # ROI/잔디 기준 시각화
        #frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = roi_color
        cv2.rectangle(frame, (self.roi_x_start - 1, self.roi_y_start - 1),
                                (self.roi_x_end + 1,   self.roi_y_end), (0, 255, 0), 1)
        cv2.circle(frame, (self.pick_x, self.pick_y), pick_rad, (111,255,111), 2)

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
        cv2.putText(frame, self.last_position_text, (10, self.roi_y_end + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 30), 2)

        if self.collecting:
            cv2.imshow('Raw Basketball Mask', raw_mask) 
        cv2.imshow('Basketball Detection', frame)
        cv2.waitKey(1)

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
                self.lost_list.clear()
                self.cx_list.clear()
                self.cy_list.clear()
                self.dis_list.clear()
                self.frame_idx = 0

                self.cam1_ball_count = 0

                self.line_start_time = time.time()

                self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames in CAM{self.cam_mode}')
        
        if self.collecting:

            self.frame_idx += 1
            self.get_logger().info(f"{self.frame_idx}")
            
            # HSV 색 조절
            self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            raw_mask = cv2.inRange(self.hsv, self.lower_hsv, self.upper_hsv) # 주황색 범위 색만
            
            # 모폴로지 연산
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

            # 컨투어
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어
            best_cnt = None
            best_ratio = 1
            x_best = y_best = None
            for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
                area = cv2.contourArea(cnt) # 1. 면적 > 1000
                if area > 1000:
                    (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                    circle_area = circle_r * circle_r * math.pi
                    ratio = abs((area / circle_area) - 1)
                    # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈
                    if ratio < best_ratio and ratio < 0.3:
                        best_ratio = ratio
                        best_cnt = cnt
                        x_best = int(x)
                        y_best = int(y)

            # 검출 결과 처리: 이전 위치 유지 로직
            if best_cnt is not None:
                # 원 탐지를 했다면
                self.lost = 0
                cx_ball = x_best + self.roi_x_start
                cy_ball = y_best + self.roi_y_start  # 전체화면에서 중심 좌표
                
                # 이전 위치 업데이트
                self.last_cx_ball = cx_ball
                self.last_cy_ball = cy_ball

                self.rect_color = (0, 255, 0)
                self.ball_color = (255, 0, 0)

            elif self.lost < 3 and self.last_cx_ball is not None:
                # 최근 10프레임 내에는 이전 위치 유지
                self.lost += 1
                cx_ball = self.last_cx_ball
                cy_ball = self.last_cy_ball
                self.ball_color = (0, 255, 255)
            else:
                # Miss 상태
                self.lost = 3
                # 표시할 위치 없음
                cx_ball = cy_ball = None
                self.rect_color = (0, 0, 255)
            
            self.frames_left -= 1
            self.lost_list.append(self.lost - 3 == 0)
            self.cx_list.append(cx_ball)
            self.cy_list.append(cy_ball)

            if self.frames_left <= 0:
                cnt = Counter(self.lost_list)
                result = max(cnt.items(), key=lambda kv: kv[1])[0]

                process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

                if result != True:
                    self.cam2_miss_count = 0

                    cxs = [a for s, a in zip(self.lost_list, self.cx_list) if s == result]
                    cys = [a for s, a in zip(self.lost_list, self.cy_list) if s == result]
                    avg_cx = int(round(np.mean(cxs)))
                    avg_cy = int(round(np.mean(cys)))

                    dx = avg_cx - self.pick_x
                    dy = avg_cy - self.pick_y

                    if math.hypot(dx, dy) <= pick_rad: # 오케이 조준 완료
                        self.cam_mode = CAM1
                        res = 9 # pick 모션

                    # elif dx, dy,,,
                        # 여기는 거리 차이에 따라서 미세조정할 세세한 값들 찾기
                        # 일단은 res = 99로 고정

                    else: 
                        res = 99

                    self.get_logger().info(f"[Ball] CAM2 Found, Relative position: {dx}, {-dy} | "
                                        f"frames= {len(self.lost_list)}, "
                                        f"wall= {process_time*1000:.1f} ms")
                    self.last_agv_cx = avg_cx
                    self.last_agv_cy = avg_cy
                    self.last_position_text = f"[Ball] Position: {avg_cx}, {avg_cy}"
                    
                    # 퍼블리시
                    msg_out = BallResult()
                    msg_out.res = res
                    self.ball_result_pub.publish(msg_out)

                else:
                    self.cam2_miss_count += 1
                    
                    if self.cam2_miss_count >= 5:
                        self.get_logger().info(f"[Ball] I totally missed,,,")
                        self.cam_mode = CAM1
                        res = 99
                        self.last_position_text = f""
                        # 퍼블리시
                        msg_out = BallResult()
                        msg_out.res = res
                        self.ball_result_pub.publish(msg_out)
                    
                    else:
                        self.get_logger().info(f"[Ball] Retry,,, | frames= {len(self.lost_list)}, "
                                            f"wall= {process_time*1000:.1f} ms")
                        res = 99
                        self.last_position_text = "Miss"
                        # 퍼블리시
                        msg_out = BallResult()
                        msg_out.res = res
                        self.ball_result_pub.publish(msg_out)

                # 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0

                self.last_position_text = ""

            # ROI랑 속도 표시
            cv2.rectangle(frame, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), self.rect_color, 1)
            cv2.circle(frame, (self.zandi_x, self.zandi_y), 3, (255, 255, 255), -1)

            # 후처리 된 마스킹 영상
            # final_mask = np.zeros_like(mask)
            # if best_cnt is not None:
            #     cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)

        else:
            self.last_position_text = "In move"

        # ROI/잔디 기준 시각화
        # frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = roi_color
        cv2.rectangle(frame, (self.roi_x_start - 1, self.roi_y_start - 1),
                                (self.roi_x_end + 1,   self.roi_y_end), (0, 255, 0), 1)
        cv2.circle(frame, (self.pick_x, self.pick_y), pick_rad, (111,255,111), 2)

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
        cv2.putText(frame, self.last_position_text, (10, self.roi_y_end + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 30), 2)

        if self.collecting:
            cv2.imshow('Raw Basketball Mask', raw_mask) # 기준 거리 이내, 주황색
        cv2.imshow('Basketball Detection', frame)
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()