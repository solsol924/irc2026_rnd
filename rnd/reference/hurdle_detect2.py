#!/usr/bin/env python3
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from rcl_interfaces.msg import SetParametersResult

import rclpy as rp
import numpy as np
import cv2
import time
import math

class HurdleDetectorNode(Node):
    def __init__(self):
        super().__init__('hurdle_detector')

        # 공용 변수
        self.image_width = 640
        self.image_height = 480

        self.roi_x_start = int(self.image_width * 1 // 5)  # 초록 박스 관심 구역
        self.roi_x_end   = int(self.image_width * 4 // 5)
        self.roi_y_start = int(self.image_height * 1 // 12)
        self.roi_y_end   = int(self.image_height * 11 // 12)

        # zandi
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머 
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_position_text = 'Dist: -- m | Pos: --, --'
        self.last_avg_text = 'AVG: --- ms | FPS: --'

        # 추적
        self.last_cx = None
        self.last_cy = None
        self.last_w = None
        self.last_h = None
        self.last_angle = None
        self.last_dis = None
        self.lost = 0

        # 변수
        self.hurdle_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)) # 가로 우대
        
        self.bridge = CvBridge()
        self.subscription_color = self.create_subscription(  # 컬러
            Image,
            '/camera/color/image_raw',  # RealSense에서 제공하는 컬러 이미지 토픽  >>  640x480 / 15fps
            self.image_callback, 10)
        
        # 퍼블리셔
        self.pub_hurdle = self.create_publisher(PointStamped, '/hurdle/position', 10)
        
        # 파라미터 선언
        self.declare_parameter("l_low", 40) # 어둡고
        self.declare_parameter("l_high", 255)  # 밝고
        self.declare_parameter("a_low", 90)  # 초록
        self.declare_parameter("a_high", 160)  # 핑크
        self.declare_parameter("b_low", 140)  # 파랑
        self.declare_parameter("b_high", 255)  # 노랑

        # 파라미터 적용
        self.l_low = self.get_parameter("l_low").value
        self.l_high = self.get_parameter("l_high").value
        self.a_low = self.get_parameter("a_low").value
        self.a_high = self.get_parameter("a_high").value
        self.b_low = self.get_parameter("b_low").value
        self.b_high = self.get_parameter("b_high").value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.lower_lab = np.array([self.l_low, self.a_low, self.b_low ], dtype=np.uint8) # 색공간 미리 선언
        self.upper_lab = np.array([self.l_high, self.a_high, self.b_high], dtype=np.uint8)
        self.lab = None      # lab 미리 선언 

        # 화면 클릭
        cv2.namedWindow('Hurdle Detection')
        cv2.setMouseCallback('Hurdle Detection', self.click)   

    def parameter_callback(self, params):
        for param in params:
            if param.name == "l_low":
                if param.value >= 0 and param.value <= self.l_high:
                    self.l_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "l_high":
                if param.value >= self.l_low and param.value <= 255:
                    self.l_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "a_low":
                if param.value >= 0 and param.value <= self.a_high:
                    self.a_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "a_high":
                if param.value >= self.a_low and param.value <= 255:
                    self.a_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "b_low":
                if param.value >= 0 and param.value <= self.b_high:
                    self.b_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "b_high":
                if param.value >= self.b_low and param.value <= 255:
                    self.b_high = param.value
                else:
                    return SetParametersResult(successful=False)
            
            self.lower_lab = np.array([self.l_low, self.a_low, self.b_low ], dtype=np.uint8) # 색공간 변하면 적용
            self.upper_lab = np.array([self.l_high, self.a_high, self.b_high], dtype=np.uint8)
            
        return SetParametersResult(successful=True)
    
    def click(self, event, x, y, _, __): # 화면 클릭
        if event != cv2.EVENT_LBUTTONDOWN or self.lab is None:
            return
        if (self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end):
            # LAB 및 BGR 읽기
            L, A, B = [int(v) for v in self.lab[y - self.roi_y_start, x - self.roi_x_start]]
            self.get_logger().info(f"[Pos] x={x - self.zandi_x}, y={- (y - self.zandi_y)} | LAB=({L},{A},{B})")
        else:
            return

    def image_callback(self, color_msg: Image):
        start_time = time.time()

        frame  = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        # ----------------------------------------------------------------------------------------------- 색보정 (CUDA X)
        # Lab 채널 이진화
        self.lab = cv2.cvtColor(roi_color, cv2.COLOR_BGR2Lab)
        mask_ab = cv2.inRange(self.lab, self.lower_lab, self.upper_lab)

        # 모폴로지 가로 9, 세로 3
        mask = cv2.morphologyEx(mask_ab, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
 
        # 컨투어 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        cx_best = cy_best = w_best = h_best = 0
        angle_best =  0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:   # 1. 너무 작은 건 제외
                (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(cnt) # 외접 사각형 그리고
                if rh > rw:
                    rw, rh = rh, rw
                    rang += 90
                aspect = rw / rh
                if aspect > 5.0:     # 2. 가로세로 비율            
                    rect_area = rw * rh
                    fill_ratio = area / rect_area
                    if 0.5 <= fill_ratio:  # 3. 면적 비율                               
                        if rw > w_best and rw > 30: # 4. 가로가 제일 긴 박스
                            best_cnt = cnt
                            cx_best, cy_best = int(rcx), int(rcy)
                            w_best, h_best = float(rw), float(rh)
                            angle_best = float(rang)

        if best_cnt is not None:
            self.lost = 0

            # 전체화면 좌표
            cx_hurdle = cx_best + self.roi_x_start
            cy_hurdle = cy_best + self.roi_y_start

            # 선 방향/법선 
            theta = math.radians(angle_best)
            dx, dy = math.cos(theta), math.sin(theta)

            # 법선 거리
            dis = abs((self.zandi_x - cx_hurdle) * -dy + (self.zandi_y - cy_hurdle) * dx)

            # 허들 직선
            x1l = int(cx_hurdle - self.image_width * dx); y1l = int(cy_hurdle - self.image_width * dy)
            x2l = int(cx_hurdle + self.image_width * dx); y2l = int(cy_hurdle + self.image_width * dy)
            cv2.line(frame, (x1l, y1l), (x2l, y2l), (0, 255, 255), 2)

            # 정보 저장
            self.last_cx = cx_best
            self.last_cy = cy_best
            self.last_w  = w_best
            self.last_h  = h_best
            self.last_angle = angle_best
            self.last_dis = dis

            self.hurdle_color = (0, 255, 0)
            self.rect_color = (0, 255, 0)
            self.last_position_text = f"Dist: {dis:.2f}px | Pos: {cx_hurdle - self.zandi_x}, {-(cy_hurdle - self.zandi_y)}, | Ang: {angle_best - 90:.2f}"

        elif self.lost < 10 and self.last_cx is not None:
            self.lost += 1
            # 정보 불러오기
            cx_hurdle = self.last_cx + self.roi_x_start
            cy_hurdle = self.last_cy + self.roi_y_start
            w_best = self.last_w
            h_best = self.last_h
            angle_best = self.last_angle
            dis = self.last_dis

            self.hurdle_color = (255, 0, 0)
            self.rect_color = (255, 0, 0)
            self.last_position_text = f"Dist: {dis:.2f}px | Pos: {cx_hurdle - self.zandi_x}, {-(cy_hurdle - self.zandi_y)}, | Ang: {angle_best - 90:.2f}"
        else:
            self.lost = 10
            cx_hurdle = cy_hurdle = None
            self.rect_color = (255, 255, 255)
            self.last_position_text = 'Miss'

        # 퍼블리시
        if cx_hurdle is not None:
            msg = PointStamped()
            msg.header = color_msg.header
            msg.point.x, msg.point.y, msg.point.z = float(cx_hurdle - self.zandi_x), float(cy_hurdle - self.zandi_y), angle_best - 90.0  
            self.pub_hurdle.publish(msg)

            # 박스 그리기
            box = cv2.boxPoints(((self.last_cx, self.last_cy), (self.last_w, self.last_h), angle_best)).astype(np.int32)
            box[:, 0] += self.roi_x_start
            box[:, 1] += self.roi_y_start
            cv2.drawContours(frame, [box], 0, self.hurdle_color, 2)
            cv2.circle(frame, (cx_hurdle, cy_hurdle), 4, (0, 0, 255), -1)

        # 화면 출력
        cv2.rectangle(frame, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), self.rect_color, 1)
        cv2.circle(frame, (self.zandi_x, self.zandi_y), 5, (255, 255, 255), -1)
        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (self.roi_x_start - 175, self.roi_y_end + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 30), 2)

        cv2.imshow('LAB Mask', mask_ab)
        cv2.imshow('Depth Mask', mask)
        cv2.imshow('Hurdle Detection', frame)
        cv2.waitKey(1)

        # FPS
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        now = time.time()
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / max(1, self.frame_count)
            fps = self.frame_count / (now - self.last_report_time)
            self.last_avg_text = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

def main():
    rp.init()
    node = HurdleDetectorNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()
