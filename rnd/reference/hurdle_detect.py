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
from robot_msgs.msg import HurdleResult, MotionEnd
from rcl_interfaces.msg import SetParametersResult

# 근접에서 사각형 각도 90도 돌아간 거
# dis값 이상한거, 하한값 넣기
# 근접에서 구간 거리 표시할까

camera_width = 640
camera_height = 480

zandi_x = int(camera_width / 2)
zandi_y = int(camera_height / 2) + 140

zandi_half_width = 150  # 잔디 중심부터 새끼발톱까지 x 거리 << 허들 박지 마라

class HurdleDetectorNode(Node):
    def __init__(self):
        super().__init__('hurdle_detector')

        self.roi_x_start = int(camera_width * 3 // 10)
        self.roi_x_end = int(camera_width * 7 // 10)
        self.roi_y_start = int(camera_height * 0 // 12)
        self.roi_y_end = int(camera_height * 12 // 12)

        # 타이머 
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.line_start_time = None

        self.last_position_text = ""
        self.last_avg_text = 'AVG: --- ms | FPS: --'

        # 추적
        self.last_cy_hurdle = self.last_left_y = self.last_right_y = None
        self.last_dy = self.last_angle = self.last_box = None
        self.hurdle_lost = 0

        # 변수
        self.hurdle_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        self.collecting = False     # 수집 중 여부
        self.armed = False               # motion_end 방어~!

        self.window_id = 0
        self.frame_idx = 0       
        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        self.collecting_frames = 15
        
        self.is_hurdle = False
        self.cy_hurdle_cal = 0
        self.jump = False

        self.is_first_decision = True

        self.first_res = True

        self.hurdle_valid_list = []  
        self.hurdle_cy_list = []
        self.hurdle_angle_list = []   
        self.hurdle_dy_list = []
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)) # 가로 9 세로 3
        
        self.bridge = CvBridge()
        self.subscription_color = self.create_subscription(  # 컬러
            Image,
            '/cam2/color/image_raw',  # RealSense에서 제공하는 컬러 이미지 토픽  >>  640x480 / 15fps
            self.cam2_image_callback, 10)
        
        self.motion_end_sub = self.create_subscription(  # 모션엔드
            MotionEnd,                   
            '/motion_end',                   
            self.motion_callback, 10) 
        
        # 퍼블리셔
        self.hurdle_result_pub = self.create_publisher(HurdleResult, '/hurdle_result', 10)
        
        # 파라미터 선언
        self.declare_parameter("hurdle_near_by", False) # 허들 가까이 왔는지
        self.declare_parameter("l_low", 40) # 어둡고
        self.declare_parameter("l_high", 255)  # 밝고
        self.declare_parameter("a_low", 90)  # 초록
        self.declare_parameter("a_high", 160)  # 핑크
        self.declare_parameter("b_low", 140)  # 파랑
        self.declare_parameter("b_high", 255)  # 노랑

        self.declare_parameter("limit_two_step", 150)  #
        self.declare_parameter("limit_one_step", 250)  #
        self.declare_parameter("limit_half_step", 300)  #
        self.declare_parameter("two_step", 200)  #
        self.declare_parameter("one_step", 100)  #
        self.declare_parameter("half_step", 50)  #

        # 파라미터 적용
        self.hurdle_near_by = self.get_parameter("hurdle_near_by").value
        self.l_low = self.get_parameter("l_low").value
        self.l_high = self.get_parameter("l_high").value
        self.a_low = self.get_parameter("a_low").value
        self.a_high = self.get_parameter("a_high").value
        self.b_low = self.get_parameter("b_low").value
        self.b_high = self.get_parameter("b_high").value

        self.limit_two_step = self.get_parameter("limit_two_step").value
        self.limit_one_step = self.get_parameter("limit_one_step").value
        self.limit_half_step = self.get_parameter("limit_half_step").value
        self.two_step = self.get_parameter("two_step").value
        self.one_step = self.get_parameter("one_step").value
        self.half_step = self.get_parameter("half_step").value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.set_roi(self.hurdle_near_by)

        self.lower_lab = np.array([self.l_low, self.a_low, self.b_low ], dtype=np.uint8) # 색공간 미리 선언
        self.upper_lab = np.array([self.l_high, self.a_high, self.b_high], dtype=np.uint8)
        self.lab = None      # lab 미리 선언 

        # 화면 클릭
        cv2.namedWindow('Hurdle Detection', cv2.WINDOW_NORMAL) ##  cv2.WINDOW_NORMAL cv2.WINDOW_NORMAL cv2.WINDOW_NORMAL cv2.WINDOW_NORMAL cv2.WINDOW_NORMAL
        cv2.setMouseCallback('Hurdle Detection', self.on_click) 

    def set_roi(self, near: bool):
        if near:
            self.roi_x_start = camera_width * 2 // 5
            self.roi_x_end   = camera_width * 3 // 5
            self.roi_y_start = camera_height * 4 // 12
            self.roi_y_end   = camera_height * 12 // 12
        else:
            self.roi_x_start = camera_width * 1 // 5
            self.roi_x_end   = camera_width * 4 // 5
            self.roi_y_start = camera_height * 0 // 12
            self.roi_y_end   = camera_height * 12 // 12  

    def parameter_callback(self, params):
        for p in params:
            if p.name == "hurdle_near_by": 
                self.hurdle_near_by = bool(p.value)
                self.set_roi(self.hurdle_near_by)
            elif p.name == "l_low": self.l_low = int(p.value)
            elif p.name == "l_high": self.l_high = int(p.value)
            elif p.name == "a_low": self.a_low = int(p.value)
            elif p.name == "a_high": self.a_high = int(p.value)
            elif p.name == "b_low": self.b_low = int(p.value)
            elif p.name == "b_high": self.b_high = int(p.value)
            elif p.name == "limit_two_step": self.limit_two_step = int(p.value)
            elif p.name == "limit_one_step": self.limit_one_step = int(p.value)
            elif p.name == "limit_half_step": self.limit_half_step = int(p.value)
            elif p.name == "two_step": self.two_step = int(p.value)
            elif p.name == "one_step": self.one_step = int(p.value)
            elif p.name == "half_step": self.half_step = int(p.value)
            
            self.lower_lab = np.array([self.l_low, self.a_low, self.b_low ], dtype=np.uint8) # 색공간 변하면 적용
            self.upper_lab = np.array([self.l_high, self.a_high, self.b_high], dtype=np.uint8)
            
        return SetParametersResult(successful=True)
    
    def on_click(self, event, x, y, _, __): # 화면 클릭
        if event != cv2.EVENT_LBUTTONDOWN or self.lab is None:
            return
        if (self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end):
            rx = x - self.roi_x_start
            ry = y - self.roi_y_start
            k = 1  # (3,3)

            y0, y1 = max(0, ry - k), min(self.roi_y_end - self.roi_y_start, ry + k + 1)
            x0, x1 = max(0, rx - k), min(self.roi_x_end - self.roi_x_start, rx + k +1)
            patch = self.lab[y0:y1, x0:x1].reshape(-1,3)
            L, A, B = np.mean(patch, axis=0).astype(int)
            self.get_logger().info(f"[Pos] x={x - zandi_x}, y={- (y - zandi_y)} | LAB=({L},{A},{B})")
        else:
            return
        
    def motion_callback(self, msg: MotionEnd): # 모션 끝 같이 받아오기 (중복 방지)
        if bool(msg.motion_end_detect):
            self.armed = True
            self.get_logger().info("Subscribed /motion_end !!!!!!!!!!!!!!!!!!!!!!!!")

    def cam2_image_callback(self, color_msg: Image):
        start_time = time.time()

        frame  = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        if not self.collecting:
            if self.armed:
                self.collecting = True
                self.armed = False
                self.frames_left = self.collecting_frames
                self.window_id += 1
                
                # 초기화
                self.hurdle_valid_list.clear()
                self.hurdle_cy_list.clear()
                self.hurdle_angle_list.clear()
                self.hurdle_dy_list.clear()
                self.frame_idx = 0

                self.last_cy_hurdle = self.last_left_y = self.last_right_y = None
                self.last_dy = self.last_angle = self.last_box = None
                self.hurdle_lost = 0

                self.line_start_time = time.time()

                self.hurdle_color = (0, 255, 0)
                self.rect_color = (0, 255, 0)

                self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames')

        if self.collecting:
            self.frame_idx += 1
            self.get_logger().info(f"step {self.frame_idx}")

            # Lab 
            self.lab = cv2.cvtColor(roi_color, cv2.COLOR_BGR2Lab)
            mask_ab = cv2.inRange(self.lab, self.lower_lab, self.upper_lab)

            # 모폴로지 가로 9, 세로 3
            mask = cv2.morphologyEx(mask_ab, cv2.MORPH_CLOSE, self.kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
    
            # 컨투어 
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_cnt = None
            best_cx_hurdle = best_cy_hurdle = best_h_hurdle = None
            best_angle = None
            best_vx = best_vy = None
            best_w_hurdle = 100
            best_ratio = 0.5

            # if self.hurdle_near_by:
            #     for cnt in contours:
            #         area = cv2.contourArea(cnt)
            #         if area > 300:
            #             (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(cnt)
            #             if rw < rh:
            #                 rw, rh = rh, rw 
            #             rect_area = rw * rh
            #             fill_ratio = area / rect_area
            #             if best_ratio < fill_ratio:
            #                 best_ratio = fill_ratio
            #                 best_cnt = cnt
            #                 best_angle = abs(rang)
            #                 best_box = cv2.boxPoints(((rcx, rcy), (rw, rh), rang)).astype(np.int32) + np.array([self.roi_x_start, self.roi_y_start])
            #                 best_dy = (zandi_y - rcy + self.roi_y_start) - rh / 2

            #     if best_cnt is not None:
            #         self.hurdle_lost = 0

            #         # 정보 저장
            #         self.last_angle = best_angle
            #         self.last_dy = best_dy
            #         self.last_box = best_box
        
            #         self.hurdle_color = (0, 255, 0)
            #         self.rect_color = (0, 255, 0)
            #         is_hurdle_valid = True

            #         cv2.drawContours(frame, [best_box], 0, self.hurdle_color, 2)

            #         self.get_logger().info(f"[Hurdle] Near! | dis= {best_dy:.1f}, angle= {best_angle:.1f}")

            #     else:
            #         if self.hurdle_lost < 3 and self.last_box is not None:
            #             self.hurdle_lost += 1

            #             # 정보 저장
            #             best_angle = self.last_angle
            #             best_dy = self.last_dy
            #             best_box = self.last_box
            
            #             self.hurdle_color = (255, 0, 0)
            #             self.rect_color = (255, 0, 0)
            #             is_hurdle_valid = True

            #             cv2.drawContours(frame, [best_box], 0, self.hurdle_color, 2)

            #             self.get_logger().info(f"[Hurdle] Lost but near! | dis= {best_dy:.1f}, angle= {best_angle:.1f}")

            #         else:
            #             self.hurdle_lost = 3

            #             best_angle = best_dy = best_box = None
            #             self.last_angle = self.last_dy = self.last_box = None                 

            #             self.rect_color = (0, 0, 255)
            #             is_hurdle_valid = False
                        
            #             self.get_logger().info(f"[Hurdle] Miss!")
                            
            # else: 
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:   # 1. 너무 작은 건 제외
                    (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(cnt) # 외접 사각형 그리고
                    if rw < rh:
                        rw, rh = rh, rw # 난 h가 짧아야 한다
                    aspect = rw / rh
                    if rw > 300:
                        if aspect > 4.0:     # 2. 가로세로 비율            
                            rect_area = rw * rh
                            fill_ratio = area / rect_area
                            if best_ratio <= fill_ratio:  # 3. 면적 비율                               
                                if rw > best_w_hurdle: # 4. 그 중에서 가로가 제일 긴 박스
                                    vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                                    if (vy < 0) or (vy == 0 and vx < 0):
                                        vx, vy = -vx, -vy
                                    vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)
                                    ang = -math.degrees(math.atan2(-vy, vx))
                                    if ang >= 90:
                                        ang -= 180  # <<< 컨투어 기준
                                    if abs(ang) < 40:
                                        best_cnt = cnt
                                        best_cx_hurdle = x0 + self.roi_x_start
                                        best_cy_hurdle = y0 + self.roi_y_start
                                        best_w_hurdle = int(rw) 
                                        best_h_hurdle = float(rh) 
                                        best_angle = float(ang)
                                        best_vx, best_vy = vx, vy
                                        best_ratio = fill_ratio

            if best_cnt is not None:
                self.hurdle_lost = 0

                # 법선 
                best_dy = (zandi_y - best_cy_hurdle) * math.cos(math.radians(best_angle)) - best_h_hurdle / 2

                # 허들 직선 표시
                left_y = int(round((self.roi_x_start - best_cx_hurdle) * best_vy / (best_vx + 0.00001) + best_cy_hurdle))
                right_y = int(round((self.roi_x_end - best_cx_hurdle) * best_vy / (best_vx + 0.00001) + best_cy_hurdle))
                cv2.line(frame, (self.roi_x_start, left_y), (self.roi_x_end, right_y), self.hurdle_color, 2)
                cv2.line(frame, (zandi_x, zandi_y), (int(round(zandi_x + math.sin(math.radians(best_angle)) * best_dy)), int(round(zandi_y - math.cos(math.radians(best_angle)) * best_dy))), (0, 0, 255), 2)
                cv2.circle(frame, (zandi_x, int(round(best_cy_hurdle))), 5, (0, 255, 0), -1)

                # 정보 저장
                self.last_cy_hurdle = best_cy_hurdle
                self.last_angle = best_angle
                self.last_left_y = left_y
                self.last_right_y = right_y
                self.last_dy = best_dy
    
                self.hurdle_color = (0, 255, 0)
                self.rect_color = (0, 255, 0)
                is_hurdle_valid = True

                self.get_logger().info(f"[Hurdle] Found! | dis= {best_dy:.1f}, angle= {best_angle:.1f}, cy= {best_cy_hurdle}")

            else:
                if self.hurdle_lost < 3 and self.last_cy_hurdle is not None:
                    self.hurdle_lost += 1
            
                    best_cy_hurdle = self.last_cy_hurdle
                    best_angle = self.last_angle
                    left_y = self.last_left_y
                    right_y = self.last_right_y
                    best_dy = self.last_dy

                    cv2.line(frame, (self.roi_x_start, left_y), (self.roi_x_end, right_y), self.hurdle_color, 2)

                    self.hurdle_color = (255, 0, 0)
                    self.rect_color = (255, 0, 0)
                    is_hurdle_valid = True
                    
                    self.get_logger().info(f"[Hurdle] Lost! | dis= {best_dy:.1f}, angle= {best_angle:.1f}, cy= {best_cy_hurdle}")

                else:
                    self.hurdle_lost = 3

                    best_cy_hurdle = best_angle = best_dy = None
                    left_y = right_y = None
                    self.last_cy_hurdle = self.last_angle = self.last_dy = None
                    self.last_left_y = self.last_right_y = None                    

                    self.rect_color = (0, 0, 255)
                    is_hurdle_valid = False
                    
                    self.get_logger().info(f"[Hurdle] Miss!")

            self.frames_left -= 1
            self.hurdle_valid_list.append(is_hurdle_valid)
            self.hurdle_cy_list.append(best_cy_hurdle)
            self.hurdle_angle_list.append(best_angle)
            self.hurdle_dy_list.append(best_dy)

            if self.frames_left <= 0:
                result = Counter(self.hurdle_valid_list).most_common(1)[0][0]
                process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0
                
                if self.jump:
                    res = 20
                    avg_angle = 0
                    self.get_logger().info(f"[Hurdle] Jump!") # 일단 임시로 # self 지정하셈
                    self.jump = False
                    self.is_hurdle = False
                    self.cy_hurdle_cal = 0
                    self.last_position_text = f"Jump!"

                else:
                    if result:
                        self.is_hurdle = True

                        center_ys = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_cy_list) if s == result and a is not None]
                        angles = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_angle_list) if s == result and a is not None]
                        delta_ys = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_dy_list) if s == result and a is not None]
                        avg_cy = int(round(np.mean(center_ys)))
                        avg_angle = np.mean(angles)
                        avg_dy = int(round(np.mean(delta_ys)))

                        if avg_cy >= self.limit_half_step:
                            res = 13 # 스텝인플레이스고고  << forward_half 임시
                            self.cy_hurdle_cal += self.half_step
                            self.jump = True
                            self.get_logger().info(f"[Hurdle] Step in place complete! Prepare to jump!")
                            self.last_position_text = f"Preparing to jump!"

                        elif avg_cy >= self.limit_one_step:
                            self.cy_hurdle_cal = avg_cy
                            if avg_angle >= 10:
                                res = 3
                                self.cy_hurdle_cal += 10
                            elif avg_angle <= -10:
                                res = 2
                                self.cy_hurdle_cal += 10
                            else:
                                res = 6 # 진짜 반발짝만
                                self.cy_hurdle_cal += self.half_step # 
                            self.get_logger().info(f"[Hurdle] Half step! | angle= {avg_angle}, dis= {avg_dy}")
                            self.last_position_text = f"angle= {avg_angle:2f}, dis= {avg_dy} | Half step"
                        
                        elif avg_cy >= self.limit_two_step: # 한 걸음
                            self.cy_hurdle_cal = avg_cy
                            if avg_angle >= 10:
                                res = 3
                                self.cy_hurdle_cal += 10
                            elif avg_angle <= -10:
                                res = 2
                                self.cy_hurdle_cal += 10
                            else:
                                res = 12
                                self.cy_hurdle_cal += self.one_step # << 한 걸음?
                            self.get_logger().info(f"[Hurdle] One step! | angle= {avg_angle}, dis= {avg_dy}")
                            self.last_position_text = f"angle= {avg_angle:2f}, dis= {avg_dy} | One step"

                        else: # 두 걸음
                            self.cy_hurdle_cal = avg_cy
                            if avg_angle >= 25: # 너무 틀어졌다 싶을 때만 방향전환
                                res = 3
                                self.cy_hurdle_cal += 10
                            elif avg_angle <= -25:
                                res = 2
                                self.cy_hurdle_cal += 10
                            else:
                                res = 27
                                self.cy_hurdle_cal += self.two_step  # << 두 걸음?
                            self.get_logger().info(f"[Hurdle] Two steps | angle= {avg_angle}, dis= {avg_dy}")
                            self.last_position_text = f"angle= {avg_angle:2f}, dis= {avg_dy} | Two steps"
                    
                    else:
                        avg_angle = 0
                        if self.is_hurdle:
                            if self.cy_hurdle_cal >= self.limit_half_step:
                                res = 13 # 스텝인플레이스고고  << forward_half 임시
                                self.cy_hurdle_cal = 0
                                self.jump = True
                                self.get_logger().info(f"[Hurdle] Step in place complete! Prepare to jump!")
                                self.last_position_text = f"Preparing to jump!"
                            elif self.cy_hurdle_cal >= self.limit_one_step: # 조금 전진 후 허들 넘기
                                res = 6 # 진짜 반스텝
                                self.cy_hurdle_cal += self.half_step
                                self.get_logger().info(f"[Hurdle] Miss! | estimated cy= {self.cy_hurdle_cal}")
                                self.last_position_text = f"Miss! | estimated cy= {self.cy_hurdle_cal} | Half step"

                            else: # 중간에 놓친 거 << 추가할 거
                                res = 12 # 그냥 걸어봐
                                self.cy_hurdle_cal += self.one_step
                                self.get_logger().info(f"[Hurdle] Miss! | estimated cy= {self.cy_hurdle_cal}")
                                self.last_position_text = f"Miss! | estimated cy= {self.cy_hurdle_cal} | One step"

                        else: # 라인
                            res = 99 # 평소
                            self.get_logger().info(f"[Hurdle] No hurdle detected")
                            self.last_position_text = f"No hurdle"
                
                if self.is_first_decision:
                    res = 99
                    self.is_first_decision = False
                    self.jump = False
                    self.is_hurdle = False
                    self.cy_hurdle_cal = 0
                    self.get_logger().info(f"[Hurdle] First decision, set to no hurdle")
                    self.last_position_text = f"First decision, no hurdle"

                # 퍼블리시
                msg_out = HurdleResult()
                msg_out.res = res
                msg_out.angle = abs(int(round(avg_angle)))
                self.hurdle_result_pub.publish(msg_out)

                self.get_logger().info(f"res= {res}")
                self.get_logger().info(f"frames= {len(self.hurdle_valid_list)}, wall= {process_time*1000:.1f} ms")

                # 리셋
                self.collecting = False
                self.frames_left = 0
                self.frame_idx = 0
            
        cv2.rectangle(frame, (self.roi_x_start + 1, self.roi_y_start + 1), (self.roi_x_end - 1, self.roi_y_end - 1), self.rect_color, 1)
        cv2.circle(frame, (zandi_x, zandi_y), 5, (255, 0, 255), -1)
        cv2.line(frame, (self.roi_x_start, zandi_y), (self.roi_x_end, zandi_y), (255, 0, 255), 2)
        
        cv2.rectangle(frame, (self.roi_x_start + 1, self.limit_one_step), (self.roi_x_end - 1, self.limit_half_step), (123, 124, 55), 2)

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
        cv2.putText(frame, self.last_position_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 30), 2)

        if self.collecting:
            cv2.imshow('LAB Mask', mask_ab)
            cv2.imshow('Mask', mask)
        cv2.imshow('Hurdle Detection', frame)
        cv2.waitKey(1)

def main(): 
    rp.init()
    node = HurdleDetectorNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()


                # if self.hurdle_near_by: # 허들 가까이 온 상태라면
                #     if result == True: # 허들을 봤다면
                #         center_ys = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_cy_list) if s == result and a is not None]
                #         angles = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_angle_list) if s == result and a is not None]
                #         delta_ys = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_dy_list) if s == result and a is not None]
                #         avg_cy = int(round(np.mean(center_ys)))
                #         avg_angle = np.mean(angles)
                #         avg_dy = int(round(np.mean(delta_ys)))

                #         if avg_dy < 10: # 박치기 임계값
                #             res = 33 # 허들 넘어라 
                #             self.hurdle_near_by = False
                #             self.hurdle_saw_once = False
                #             self.hurdle_never_seen = True
                #             self.get_logger().info(f"[Hurdle] GOGOGOGO | angle= {avg_angle}, dis= {avg_dy}")
                            
                #             self.set_roi(self.hurdle_near_by)
                #         else:
                #             res = 44 # 거의 제자리 걸음으로 접근
                #             self.get_logger().info(f"[Hurdle] Parking,,, | angle= {avg_angle}, dis= {avg_dy}")

                #     else: # 허들을 못 봐도
                #         res = 44 # 일단 걸어봐
                #         avg_angle = 0
                #         self.get_logger().info(f"[Hurdle] Parking,,,")

                # else:
                #     if result == True: # 허들을 봤다면
                #         self.hurdle_miss_count = 0

                #         if self.hurdle_never_seen: # 여긴 그냥 지금껏 허들을 본 적 있는지 없는지 저장용
                #             self.hurdle_saw_once = True
                #             self.hurdle_never_seen = False

                #         angles = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_angle_list) if s == result and a is not None]
                #         delta_ys = [a for s, a in zip(self.hurdle_valid_list, self.hurdle_dy_list) if s == result and a is not None]
                #         avg_angle = np.mean(angles)
                #         avg_dy = int(round(np.mean(delta_ys)))

                #         dis_min = avg_dy - zandi_half_width * math.cos(math.radians(avg_angle))

                #         if dis_min >= 250: # 1구간
                #             if avg_angle >= 15:
                #                 res = 3
                #             elif avg_angle <= -15:
                #                 res = 2
                #             else:
                #                 res = 12
                #             self.get_logger().info(f"[Hurdle] Approaching 1 | angle= {avg_angle}, dis= {avg_dy}")

                #         elif dis_min >= 100: # 2구간
                #             if avg_angle >= 8:
                #                 res = 3
                #             elif avg_angle <= -8:
                #                 res = 2
                #             else:
                #                 res = 12
                #             self.get_logger().info(f"[Hurdle] Approaching 2 | angle= {avg_angle}, dis= {avg_dy}")

                #         else: # 너 좀 가깝다
                #             self.hurdle_near_by = True
                #             self.set_roi(self.hurdle_near_by)

                #             if avg_angle >= 8:
                #                 res = 3
                #             elif avg_angle <= -8:
                #                 res = 2
                #             else:
                #                 res = 12
                #             self.get_logger().info(f"[Hurdle] Near by! | angle= {avg_angle}, dis= {avg_dy}")

                #     else: # 허들을 못 봤다면
                #         avg_angle = 0
                #         if self.hurdle_saw_once: # 허들 한 번이라도 봤으면 무조건 있는 거임 ㅇㅇ~
                #             self.hurdle_miss_count += 1
                #             if self.hurdle_miss_count >= 2:
                #                 res = 44 # 색공간 넓히거나, 조건 수정 필요
                #                 self.get_logger().info(f"[Hurdle] Totally lost,,,")
                #             else: 
                #                 res = 12
                #                 self.get_logger().info(f"[Hurdle] Trying to find,,,")
                #         else:
                #             res = 99 # 평소
                #             self.get_logger().info(f"[Hurdle] No hurdle")