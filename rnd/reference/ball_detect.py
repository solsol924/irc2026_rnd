#!/usr/bin/env python3
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from geometry_msgs.msg import PointStamped  # type: ignore
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer # type: ignore  동기화
from rcl_interfaces.msg import SetParametersResult

import rclpy  # type: ignore
import cv2
import numpy as np
import time

class BasketballDetectorNode(Node):
    def __init__(self):
        super().__init__('basketball_detector')
        
        # 공용 변수
        self.image_width = 640  #카메라 영상의 기본 해상도 640x480 설정
        self.image_height = 480

        self.roi_x_start = int(self.image_width * 1 // 5)  # 화면 내 초록 박스 관심 구역
        self.roi_x_end   = int(self.image_width * 4 // 5)   # 전체 화면의 1/5 ~ 4/5 사이를 관심 영역으로 설정
        self.roi_y_start = int(self.image_height * 1 // 12)  # 세로 방향으로는 화면 상단에서 1/12 ~ 11/12 사이를 관심 영역으로 설정
        self.roi_y_end   = int(self.image_height * 11 // 12)

        # zandi
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = 'AVG: --- ms | FPS: --'
        self.last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력

        # 추적
        self.last_cx_img = None       #바로 직전 화면에서의 농구공 중심 좌표 
        self.last_cy_ball = None
        self.last_radius = None  
        self.last_z = None           #카메라로부터의 깊이 거리
        self.lost = 0                #농구공을 찾지 못한 횟수 

        # 변수
        self.ball_color = (0, 255, 0) # 검출된 공의 원 색상 (초록색)
        self.rect_color = (0, 255, 0) # ROI 사각형 색상 (초록색)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # 모폴로지 연산용 커널 (5x5 타원형)

        self.fx, self.fy = 607.0, 606.0   # ros2 topic echo /camera/color/camera_info - 카메라 고유값
        self.cx_intr, self.cy_intr = 325.5, 239.4  # 카메라 렌즈의 실제 중심이 이미지 센서에 맺히는 좌표 (화면 중앙과 렌즈의 중앙 달라서)

        self.lower_hsv = np.array([8, 60, 60])  # 색의 범위 
        self.upper_hsv = np.array([60, 255, 255]) # 주황색 기준으로
        self.depth_thresh = 1500.0  # mm 기준 마스크 거리  # 1.5m 이상 물체는 무시 
        self.depth_scale = 0.001    # m 변환용  mm를 m로 바꿔주는 스케일
        
        self.bridge = CvBridge() # ros 이미지 메시지를 OpenCV 이미지로 변환하기 위한 브릿지

        # 컬러, 깊이 영상 동기화
        color_sub = Subscriber(self, Image, '/camera/color/image_raw') # 칼라
        depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw') # 깊이
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.image_callback) # 컬러 + 깊이 영상이 동시에 들어오면 image_callback 함수 실행

        # 파라미터 선언 디테일하게 설정 - HSV 범위 조절을 위한 파라미터 선언
        self.declare_parameter("h_low", 8) 
        self.declare_parameter("h_high", 60)  # 8~60 사이의 Hue 값이 주황색 범위로 설정
        self.declare_parameter("s_low", 60)  
        self.declare_parameter("s_high", 255)  # 채도 범위 60~255로 설정 (낮은 채도는 제외) 흐리멍덩한 색이나 회색빛 도는 물체 무시 
        self.declare_parameter("v_low", 0)  
        self.declare_parameter("v_high", 255)  # 명도 범위 0~255로 설정 (모든 명도 허용) 밝기 제한은 두지 않음, 어두운 환경에서도 검출 가능하도록

        # 파라미터 적용
        self.h_low = self.get_parameter("h_low").value # HSV 범위 초기값을 파라미터에서 읽어와서 변수에 저장
        self.h_high = self.get_parameter("h_high").value
        self.s_low = self.get_parameter("s_low").value
        self.s_high = self.get_parameter("s_high").value
        self.v_low = self.get_parameter("v_low").value
        self.v_high = self.get_parameter("v_high").value

        self.add_on_set_parameters_callback(self.parameter_callback) #변화를 감지해서 적용하는 콜백 등록

        self.lower_hsv = np.array([self.h_low, self.s_low, self.v_low ], dtype=np.uint8) # 색공간 미리 선언
        self.upper_hsv = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8) # 수학적인 묶음으로 HSV 범위를 배열로 만들어서 나중에 마스크 생성할 때 사용하기 편하게 함
        self.hsv = None      # hsv 미리 선언 - 화면 클릭 시 HSV 값 읽어오기 위해 빈 그릇 미리 준비 

        self.pub_ball = self.create_publisher(PointStamped, '/basketball/position', 10) # 나중에 다른 노드에 전송하기 위한 퍼블리셔

        # 화면 클릭
        cv2.namedWindow('Basketball Detection') # OpenCV 창 이름 설정 - 나중에 이 창에서 마우스 클릭 이벤트를 감지하기 위해 필요
        cv2.setMouseCallback('Basketball Detection', self.click)  # 마우스 클릭 이벤트가 발생하면 self.click 함수가 호출되도록 설정

    def parameter_callback(self, params): #외부에서 파라미터 변경 시 유효성 검사 및 적용하는 콜백 함수
        for param in params:
            if param.name == "h_low":
                if param.value >= 0 and param.value <= self.h_high:
                    self.h_low = param.value #조건에 맞으면 변수에 적용
                else:
                    return SetParametersResult(successful=False) #조건에 맞지 않으면 파라미터 변경 실패 응답 반환
            if param.name == "h_high":
                if param.value >= self.h_low and param.value <= 255:
                    self.h_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "s_low":
                if param.value >= 0 and param.value <= self.s_high:
                    self.s_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "s_high":
                if param.value >= self.s_low and param.value <= 255:
                    self.s_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "v_low":
                if param.value >= 0 and param.value <= self.v_high:
                    self.v_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "v_high":
                if param.value >= self.v_low and param.value <= 255:
                    self.v_high = param.value
                else:
                    return SetParametersResult(successful=False)
            
            self.lower_hsv = np.array([self.h_low, self.s_low, self.v_low ], dtype=np.uint8) # 색공간 변하면 적용
            self.upper_hsv = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8) # 3개의 변수 하나로 합치기 
            
        return SetParametersResult(successful=True) # 모든 파라미터가 유효하면 변경 성공 응답 반환
    
    def click(self, event, x, y, _, __): # 화면 클릭
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None: # 왼쪽 버튼 클릭이 아니거나 HSV 이미지가 준비되지 않았으면
            return
        if (self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end):
            # LAB 및 BGR 읽기 # 클릭한 위치가 ROI 안에 있으면 계산 진행 
            H, S, V = [int(v) for v in self.hsv[y - self.roi_y_start, x - self.roi_x_start]] 
            self.get_logger().info(f"[Pos] x={x - self.zandi_x}, y={- (y - self.zandi_y)} | HSV=({H},{S},{V})") # 터미널 창에 문자로 출력 좌우로 몇 픽셀 떨어져 있는지, HSV 값이 무엇인지 출력
        else:
            return

    def image_callback(self, color_msg: Image, depth_msg: Image):
        start_time = time.time()
        
        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        roi_depth = depth[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        # HSV 색 조절
        self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(self.hsv, self.lower_hsv, self.upper_hsv) # 주황색 범위 색만 마스크로 추출 - 기준 거리 이내, 주황색 영역 흰색, 나머지 검은색
        raw_mask[roi_depth >= self.depth_thresh] = 0  # 깊이 기준으로 멀리 있는 픽셀은 마스크에서 제외 - 기준 거리 이상인 부분은 마스크에서 제거 (검은색으로)

        mask = raw_mask.copy() # 출력용 raw_mask

        # 모폴로지 연산
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

        # 컨투어
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어
        best_cnt = None
        best_ratio = 1
        x_best = y_best = radius = None
        for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
            area = cv2.contourArea(cnt) # 1. 면적 > 1000 
            if area > 1000:
                (x, y), circle_r = cv2.minEnclosingCircle(cnt) #가장 작은 원을 그리고 외접하는 원의 중심 좌표와 반지름 계산
                circle_area = circle_r * circle_r * 3.1416 # 가상의 원 넓이 구하기 
                ratio = abs((area / circle_area) - 1) # 컨투어 면적과 외접원 면적의 비율이 1에 가까울수록 원형에 가깝다고 판단, 이 비율이 가장 작은 컨투어를 최종 후보로 선택
                # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈
                if ratio < best_ratio and ratio < 0.3: #동그라미 오차율 30%미만인 것들만 취급 
                    best_ratio = ratio
                    best_cnt = cnt
                    x_best = int(x)
                    y_best = int(y)
                    radius = circle_r

        # 검출 결과 처리: 이전 위치 유지 로직
        if best_cnt is not None:
            # 원 탐지를 했다면
            self.lost = 0 #  공을 찾았으니 lost 초기화
            cx_ball = x_best + self.roi_x_start
            cy_ball = y_best + self.roi_y_start  # 전체화면에서 중심 좌표

            # 깊이 계산 - 원 중심을 기준으로 3x3 영역의 평균 깊이값을 사용하여 z 계산, 이렇게 하면 노이즈에 좀 더 강해짐
            x1 = max(x_best - 1, 0)
            x2 = min(x_best + 2, self.image_width)
            y1 = max(y_best - 1, 0)
            y2 = min(y_best + 2, self.image_height)

            roi_patch = roi_depth[y1:y2, x1:x2] # x영역 y영역 3x3으로 잘라서 roi_patch에 저장
            z = float(roi_patch.mean()) * self.depth_scale # 9개의 픽셀의 평균 깊이값을 계산해서 z에 저장, mm를 m로 변환하기 위해 depth_scale 곱하기
            
            # 이전 위치 업데이트 (농구공이 갑자기 사라졌을 때 이전 위치를 최대 10프레임까지 유지하기 위해)
            # 플랜 A: 공을 찾았으니 현재 위치로 업데이트
            self.last_cx_img = cx_ball
            self.last_cy_ball = cy_ball
            self.last_radius = radius
            self.last_z = z

            self.rect_color = (0, 255, 0) # ROI 초록색으로 표시 - 공을 찾았으니 관심 영역도 초록색으로 표시
            self.ball_color = (255, 0, 0) # 공을 찾은 경우는 파란색으로 표시

        elif self.lost < 10 and self.last_cx_img is not None: # 공을 못 찾았지만, 최근 10프레임 내에는 공을 찾았던 적이 있다면
            # 최근 10프레임 내에는 이전 위치 유지
            # 플랜 B: 공을 못 찾았지만 최근 위치 유지 (최대 10프레임)
            self.lost += 1
            cx_ball = self.last_cx_img
            cy_ball = self.last_cy_ball
            radius = self.last_radius
            z = self.last_z
            self.ball_color = (0, 255, 255) # 공을 못 찾았지만 최근 위치 유지하는 경우는 노란색으로 표시

            # 플랜 C: 공을 못 찾은 지 10프레임이 넘었다면, 위치 정보 초기화
        else:
            # Miss 상태 
            self.lost = 10 # 카운트 10으로 고정해서 계속 Miss 상태 유지
            self.last_position_text = 'Miss'
            # 표시할 위치 없음
            cx_ball = cy_ball = radius = z = None
            self.rect_color = (0, 0, 255) # ROI 빨간색으로 표시 - 공을 못 찾았으니 관심 영역도 빨간색으로 표시

        # 위치 퍼블리시 및 화면 표시
        if cx_ball is not None:
            # 카메라까지 거리 보정 값
            # 2D 이미지 좌표 (cx_ball, cy_ball)와 깊이 z를 이용해서 3D 공간에서의 X, Y, Z 좌표 계산
            X = (cx_ball - self.cx_intr) * z / self.fx
            Y = (cy_ball - self.cy_intr) * z / self.fy

            # 로봇의 다른 시스템에 위치 정보 전송 
            msg = PointStamped() # 메세지 만들고 (점의 좌표와 시간 정보 담는 메시지)
            msg.header = color_msg.header # 칼라 화면 프레임 정보
            msg.point.x, msg.point.y, msg.point.z = X, Y, z # 공 위치 좌표 3개
            self.pub_ball.publish(msg) # 보내기

            # 원 그리기
            cv2.circle(frame, (cx_ball, cy_ball), int(radius), self.ball_color, 2) # 정상일 때는 파란색, 놓치지 직전에는 노란색으로 원 그리기
            cv2.circle(frame, (cx_ball, cy_ball), 5, (0, 0, 255), -1) # 공의 중심에 빨간 점 그리기 

            # 원점과의 거리 정보
            dx = cx_ball - self.zandi_x
            dy = cy_ball - self.zandi_y
            self.last_position_text = f'Dist: {z:.2f}m | Pos: {dx}, {-dy}'
 
        # ROI랑 속도 표시
        cv2.rectangle(frame, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), self.rect_color, 1)
        cv2.circle(frame, (self.zandi_x, self.zandi_y), 3, (255, 255, 255), -1) # 잔디 위치에 흰색 점 그리기

        # 후처리 된 마스킹 영상
        final_mask = np.zeros_like(mask) # 최종적으로 공으로 판단한 영역만 흰색으로 표시하기 위한 까만 빈 마스크 만들기
        if best_cnt is not None:
            cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED) # 최종 후보 컨투어 영역을 최종 마스크에 흰색으로 채워서 그리기, 이렇게 하면 최종적으로 공으로 판단한 영역만 흰색으로 표시된 마스크가 만들어짐

        # 딜레이 측정
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        now = time.time()
        if now - self.last_report_time >= 1.0: # 1초마다 평균 처리 시간과 FPS 계산해서 화면에 표시하기
            avg_time = self.total_time / self.frame_count # 사진 한장을 처리하는데 걸리는 평균 시간 계산하기
            fps = self.frame_count / (now - self.last_report_time) # 초당 몇 프레임 처리했는지 계산하기 (숫자가 높을수록 빠르게 처리했다는 뜻)
            self.last_avg_text = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (10, self.roi_y_end + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 30), 2)

        cv2.imshow('Raw', raw_mask) # 기준 거리 이내, 주황색
        cv2.imshow('Mask', mask) # 기준 거리 이내, 주황색, 보정 들어간 마스크
        cv2.imshow('Final Mask', final_mask) # 최종적으로 공이라 판단한 마스크
        cv2.imshow('Basketball Detection', frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = BasketballDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
