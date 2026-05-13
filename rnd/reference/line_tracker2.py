
# 최종 합본

import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque, Counter
from my_cv_msgs.msg import LinePoint, LinePointsArray # type: ignore
from rcl_interfaces.msg import SetParametersResult

class RectangleTracker:
    def __init__(self, max_lost, max_dist, min_found):
        self.rectangles = {}     # {id: (cx, cy, lost, found)}
        self.next_id = 0
        self.max_lost = max_lost
        self.max_dist = max_dist
        self.min_found = min_found

    def update(self, detected_centroids):
        matched_ids = set()
        updated_rectangles = {}

        for cx, cy in detected_centroids: # 1. 이번 프레임에 감지된 중심점 리스트랑 기존에 있던 중심점 리스트랑 매칭
            best_id = None
            best_dist = self.max_dist

            for id, (prev_cx, prev_cy, lost, found) in self.rectangles.items(): # 1-1. 기존에 있던 리스트 하나씩 다 돌리는 중,,,,,,,
                if id in matched_ids:
                    continue  # 매칭된 점 있으면 그 루프는 건너뜀
                dist = math.hypot(cx - prev_cx, cy - prev_cy) # 거리 비교
                if dist < best_dist: # 가장 가까운 점 찾기
                    best_dist = dist
                    best_id = id

            if best_id is not None: # 1-2. 원래 있던 점 찾았다 >>> 좌표 갱신
                _, _, _, found = self.rectangles[best_id]
                updated_rectangles[best_id] = (cx, cy, 0, min(found + 1, self.min_found)) # found값 계속 +1 (최대 min_found)
                matched_ids.add(best_id)
            else: # 1-3. 원래 없던 점 찾았다 >>> 좌표랑 아이디 입력
                updated_rectangles[self.next_id] = (cx, cy, 0, 0)
                self.next_id += 1

        # 2. 이번 프레임에 감지 못 한 기존에 있던 중심점 처리
        for id, (cx, cy, lost, found) in self.rectangles.items():
            if id not in matched_ids:
                lost += 1 # lost 늘리고
                if lost < self.max_lost: # 2-1. 조건 만족하면 좌표랑 다른 거 다 그대로 유지
                    updated_rectangles[id] = (cx, cy, lost, found) 
                # else:  >>>  2-2. lost가 max_lost 이상이면 자동 삭제

        self.rectangles = updated_rectangles # 기존 중심점 리스트 갱신

        # 3. 0 <= lost < self.max_lost 이고, found >= min_found 인 애들만 쓸만한 사각형으로 간주, 리턴
        valid_rects = {}
        for id, (cx, cy, lost, found) in self.rectangles.items():
            if found >= self.min_found: # lost 많은 애들은 이미 위에서 컷
                valid_rects[id] = (cx, cy, lost, found)

        return valid_rects

class LineListenerNode(Node):
    def __init__(self):
        super().__init__('img_subscriber')                          
        self.subscription_color = self.create_subscription(  # 이미지 토픽
            Image,
            '/camera/color/image_raw',  #  640x480 / 15fps
            self.color_image_callback, 10)
        
        self.bridge = CvBridge()

        # 파라미터 선언
        self.declare_parameter("max_lost", 10)
        self.declare_parameter("max_dist", 50)
        self.declare_parameter("min_found", 10)

        self.declare_parameter("h_up", 180)
        self.declare_parameter("h_down", 0)
        self.declare_parameter("s_up", 40)
        self.declare_parameter("s_down", 0)
        self.declare_parameter("v_up", 255)
        self.declare_parameter("v_down", 70)

        self.declare_parameter("epsilon_n", 0.08)
        self.declare_parameter("rect_area_max", 3000)
        self.declare_parameter("rect_area_min", 500)

        self.declare_parameter("max_len", 30)
        self.declare_parameter("delta_s", 15)
        self.declare_parameter("vertical", 75)
        self.declare_parameter("horizonal", 15)

        # 파라미터 적용
        max_lost = self.get_parameter("max_lost").value
        max_dist = self.get_parameter("max_dist").value
        min_found = self.get_parameter("min_found").value
        
        self.h_up = self.get_parameter("h_up").value
        self.h_down = self.get_parameter("h_down").value
        self.s_up = self.get_parameter("s_up").value
        self.s_down = self.get_parameter("s_down").value
        self.v_up = self.get_parameter("v_up").value
        self.v_down = self.get_parameter("v_down").value

        self.epsilon_n = self.get_parameter("epsilon_n").value
        self.rect_area_max = self.get_parameter("rect_area_max").value
        self.rect_area_min = self.get_parameter("rect_area_min").value

        self.max_len = self.get_parameter("max_len").value
        self.delta_s = self.get_parameter("delta_s").value
        self.vertical = self.get_parameter("vertical").value
        self.horizonal = self.get_parameter("horizonal").value
        
        # 기타
        self.curve_count = 0
        self.tilt_text = "" # 화면 출력
        self.curve_text = "" # 화면 출력
        self.recent_curve = deque(maxlen=self.max_len) # maxlen 프레임 중에서 최빈값
        self.recent_tilt = deque(maxlen=self.max_len)

        self.add_on_set_parameters_callback(self.parameter_callback)

        # 시간 디버깅
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"

        self.tracker = RectangleTracker(max_lost, max_dist, min_found) # 신규 10프레임 실종 10프레임 / 50픽셀

    def parameter_callback(self, params):
        for param in params:
            if param.name == "max_lost":
                if param.value > 0:
                    self.max_lost = param.value
                    self.tracker.max_lost = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "max_dist":
                if param.value > 0:
                    self.max_dist = param.value
                    self.tracker.max_dist = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "min_found":
                if param.value > 0:
                    self.min_found = param.value
                    self.tracker.min_found = param.value
                else:
                    return SetParametersResult(successful=False)

            elif param.name == "h_up":
                if param.value >= self.h_down and param.value <= 255:
                    self.h_up = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "h_down":
                if param.value <= self.h_up and param.value >= 0:
                    self.h_down = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "s_up":
                if param.value >= self.s_down and param.value <= 255:
                    self.s_up = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "s_down":
                if param.value <= self.s_up and param.value >= 0:
                    self.s_down = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "v_up":
                if param.value >= self.v_down and param.value <= 255:
                    self.v_up = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "v_down":
                if param.value <= self.v_up and param.value >= 0:
                    self.v_down = param.value
                else:
                    return SetParametersResult(successful=False)

            elif param.name == "epsilon_n":
                if param.value < 1.00 and param.value >= 0.01:
                    self.epsilon_n = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "rect_area_max":
                if param.value > self.rect_area_min:
                    self.rect_area_max = param.value
                else:
                    return SetParametersResult(successful=False)
            elif param.name == "rect_area_min":
                if param.value < self.rect_area_max and param.value >= 0:
                    self.rect_area_min = param.value
                else:
                    return SetParametersResult(successful=False)
                
            elif param.name == "max_len":
                if param.value > 0:
                    self.max_len = param.value
                    self.recent_curve = deque(maxlen=self.max_len)
                    self.recent_tilt = deque(maxlen=self.max_len)
                else:
                    return SetParametersResult(successful=False)
            if param.name == "delta_s":
                if param.value > 0:
                    self.delta_s = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "vertical":
                if param.value > self.horizonal and param.value < 90:
                    self.vertical = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "horizonal":
                if param.value > 0 and param.value < self.vertical:
                    self.horizonal = param.value
                else:
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

    def get_angle(self, c1, c2): # 단순 각도 계산
        dx = c2[0]-c1[0]
        dy = c2[1]-c1[1]
        angle = 180 / math.pi * math.atan2(dy, dx) + 90
        if angle > 89:  # 계산 과부하 방지
            angle = 89
        elif angle < -89:
            angle = -89
        return round(angle, 2)

    def color_image_callback(self, msg): # 이미지 받아오기

        start_time = time.time()  # 시작 시간 기록

        # ROS 이미지 CV 이미지로 받아오기
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        screen_h, screen_w, _ = cv_image.shape

        # ROI 좌표 정의, 컷
        roi_y_start = screen_h * 1 // 3  # 위
        roi_y_end = screen_h // 1        # 아래
        roi_x_start = screen_w * 2 // 5  # 왼 
        roi_x_end = screen_w * 3 // 5    # 오

        roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # GPU 메모리에 이미지 업로드   ----------------------------------------------------------------------------------------------CUDA 시작 / 보정 시작
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(roi)

        # CUDA용 Gaussian 필터 생성
        gpu_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 0)

        # Gaussian 필터 적용
        gpu_blurred = gpu_gaussian.apply(gpu_image)

        # BGR -> HSV 변환 (CUDA 버전)
        gpu_hsv = cv2.cuda.cvtColor(gpu_blurred, cv2.COLOR_BGR2HSV)

        # GPU 메모리에서 HSV 이미지 다운로드   ---------------------------------------------------------------------------------------CUDA 끝
        hsv = gpu_hsv.download()   

        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        h, s, v = cv2.split(hsv) 
        v = clahe.apply(v) # v값만 보정
        hsv_image = cv2.merge((h, s, v))
        
        #--------------------------------------------------------------------------------------------------------------------- 보정 끝

        # 1. BGR 이미지로 변환
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # 보정 끝낸 bgr 화면 < 내가 보려고 띄우는 화면

        # 2. HSV 흰색 범위 지정
        hsv_upper_white = np.array([self.h_up, self.s_up, self.v_up])
        hsv_lower_white = np.array([self.h_down, self.s_down, self.v_down])
        hsv_mask = cv2.inRange(hsv_image, hsv_lower_white, hsv_upper_white) # 이 화면으로 판단

        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_centroids = []  # 무게중심 리스트 (cx, cy)
        
        for cnt in contours:     # 사각형 찾기
            epsilon = self.epsilon_n * cv2.arcLength(cnt, True)  # 근사화 정확도 낮춤 / 0.05정도로 하라는데 조정 필요
            approx = cv2.approxPolyDP(cnt, epsilon, True)  # 윤곽선 근사

            if len(approx) == 4:  # 1. 모서리가 4개면,
                rect = cv2.minAreaRect(approx) # 윤곽선 최소 외접 사각형(박스)
                (box_cx, box_cy), (box_w, box_h), box_ang = rect
                box_cx, box_cy = int(box_cx), int(box_cy)
                box_w, box_h = int(box_w), int(box_h)
                area_rect = box_w * box_h
                if self.rect_area_max > area_rect > self.rect_area_min:  # 2. 외접 박스 면적이 500 이상, 3000이하면,
                    angles = []
                    for i in range(4):  # 각도 계산  >>>>>>>>> 이거 연산이 좀 많긴 함 
                        p1, p2, p3 = approx[i - 1][0], approx[i][0], approx[(i + 1) % 4][0]
                        v1 = np.array(p1) - np.array(p2)
                        v2 = np.array(p3) - np.array(p2)
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # 라디안 -> 도 변환
                        angles.append(angle)
                    if all(60 <= angle <= 120 for angle in angles): # 3. 내각이 60~120 사이면,
                        cv2.drawContours(bgr_image, [approx], 0, (200, 0, 255), 2)  # 테두리 핑크 / 사각형 검출 하는지 확인용 / 지금 프레임에서 사각형이라고 검출된 모든 사각형
                        rect_centroids.append((box_cx,box_cy))  # 사각형 중심 리스트에 추가 
                            
        # -------------------------------------------------------------------- 사각형 추적

        valid_rects = self.tracker.update(rect_centroids)  # 사각형 중심을 추적기에 보냄 {id: (cx, cy, lost, found)} / lost & found 조건도 만족한 사각형            
        for id,(cx,cy,lost,found) in valid_rects.items(): 
            cv2.putText(bgr_image, f"ID: {id}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # 유효 사각형은 id 출력

        candidates = sorted(
            [(cx, cy, lost) for (cx, cy, lost, found) in valid_rects.values()],
            key=lambda c: c[1], reverse=True)[:5] # 후보점 5개 선택 (화면 아래쪽부터)

        #---------------------------------------------------------------라인 판단 시작  << 최소자승 직선 근사  << 얘 좀 이상함
            
        if len(candidates) >= 3:  # 1. 3개 이상일 때: 맨 위 점 제외하고 직선 근사        
            tip_x, tip_y, tip_lost = candidates[-1] # 그 중 가장 위의 점 = 새로 탐지된 점
            if tip_lost > 0:
                cv2.circle(roi, (tip_x, tip_y), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
            else:
                cv2.circle(roi, (tip_x, tip_y), 4, (0, 0, 255), -1) # 가장 위의 새로운 점은 빨강
            
            for acx, acy, ac_lost in candidates[:-1]:
                if ac_lost > 0:
                    cv2.circle(roi, (acx, acy), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
                else:
                    cv2.circle(roi, (acx, acy), 4, (255, 0, 0), -1) # 나머지 파랑

            xs = np.array([c[0] for c in candidates[:-1]])
            ys = np.array([c[1] for c in candidates[:-1]])

            if abs(xs.max() - xs.min()) < 2:  # 점들이 수직에 가까운 경우
                x1, y1, _ = candidates[0]
                x2, y2, _ = candidates[-2]
                y1 = roi_y_end - roi_y_start
                y2 = 0
                line_angle = 90
                delta = (xs.max() + xs.min()) / 2 - tip_x
                # 커브 판단
                if delta < -1 * self.delta_s:
                    self.curve_text = "Turn Right"
                elif delta > self.delta_s:
                    self.curve_text = "Turn Left"
                else:
                    self.curve_text = "Straight"
                self.tilt_text = "Straight" # 기울기 판단 > 수직이니까 항상 직진

            else: # 일반적인 경우
                m, b = np.polyfit(xs, ys, 1) # 기울기와 y절편
                line_angle = math.degrees(math.atan2(m, 1))
                numerator = m * tip_x - tip_y + b # 점과 직선 사이의 수직 거리
                denominator = math.sqrt(m**2 + 1)
                delta = numerator / denominator
                # 직선 시각화 방식 선택 (기울기 크기에 따라)
                if abs(m) > 1:  # 기울기 크면 y 기준 (거의 수직)
                    y1 = roi_y_end - roi_y_start
                    y2 = 0
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:  # 거의 수평일 땐 x 기준
                    x1 = roi_x_end - roi_x_start
                    x2 = 0
                    y1 = int(m * (x1) + b)
                    y2 = int(m * (x2) + b)
                # 커브 판단
                if delta < -1 * self.delta_s:
                    if m > 0:
                        self.curve_text = "Turn Left"
                    elif m < 0:
                        self.curve_text = "Turn Right"
                    else:
                        self.curve_text = "Horizon" # 예) 농구공 집고 라인을 찾는데 라인이 수평선인 경우 -> 별도의 알고리즘 필요 - 추후에,,,
                elif delta > self.delta_s:
                    if m > 0:
                        self.curve_text = "Turn Right"
                    elif m < 0:
                        self.curve_text = "Turn Left"
                    else:
                        self.curve_text = "Horizon" ###
                else:
                    self.curve_text = "Straight"
                # 기울기 판단
                if self.horizonal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizonal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else: # 평행선 아직 처리 안 함(0~10 사이)
                    self.tilt_text = "Straight"
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

        elif len(candidates) == 2:  # 2. 2개일 때: 걍 두 개 잇기 >>> 여기부턴 그럴 일 거의 없다고 봄
            x1, y1, down_lost = candidates[0] # 아래
            x2, y2, up_lost = candidates[1] # 위
            if down_lost > 0:
                cv2.circle(roi, (x1, y1), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
            else:
                cv2.circle(roi, (x1, y1), 4, (0, 0, 255), -1) #  파랑
            if up_lost > 0:
                cv2.circle(roi, (x2, y2), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
            else:
                cv2.circle(roi, (x2, y2), 4, (255, 0, 0), -1) # 빨강

            if abs(x1 - x2) < 2: # 점들이 수직에 가까운 경우
                line_angle = 90
                self.tilt_text = "Straight"
                y1 = roi_y_end - roi_y_start
                y2 = 0
            else:   # 일반적인 경우
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                line_angle = math.degrees(math.atan2(m, 1))
                if abs(m) > 1:  # 기울기 크면 y 기준 (거의 수직)
                    y1 = roi_y_end - roi_y_start
                    y2 = 0 
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:  # 거의 수평일 땐 x 기준
                    x1 = roi_x_end - roi_x_start
                    x2 = 0
                    y1 = int(m * x1 + b)
                    y2 = int(m * x2 + b)
                
                if self.horizonal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizonal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else: # 평행성 아직 처리 안 함
                    self.tilt_text = "Straight"
            self.curve_text = "Straight" # 커브 판단 불가
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

        else: # 감지 실패
            self.curve_text = "Miss"
            self.tilt_text = "Miss"
            line_angle = 0

        self.recent_curve.append(self.curve_text)
        self.recent_tilt.append(self.tilt_text)
        stable_curve = Counter(self.recent_curve).most_common(1)[0][0] if self.recent_curve else "Miss"  # 최빈값에 맞게 커브 판단  >> 후에 이거로 상태함수 넣어도 됨 / fsm
        stable_tilt = Counter(self.recent_tilt).most_common(1)[0][0] if self.recent_tilt else "Miss"

        #---------------------------------------------------------------------------------------------------------------라인 판단 끝

        # 방향 텍스트 출력
        cv2.putText(cv_image, f"Rotate: {line_angle:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Tilt: {stable_tilt}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Curve: {stable_curve}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = roi  # 전체화면
        cv2.rectangle(cv_image, (roi_x_start - 1, roi_y_start - 1), (roi_x_end + 1, roi_y_end), (0, 255, 0), 1) # ROI 구역 표시
        
        #-------------------------------------------------------------------------------------------------- 프레임 처리 시간 측정

        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed

        now = time.time()

        # 1초에 한 번 평균 계산
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            avg_fps = self.frame_count / (now - self.last_report_time)
            
            # 텍스트 준비
            self.last_avg_text = f"PING: {avg_time*1000:.2f}ms | FPS: {avg_fps:.2f}"
            
            # 타이머 리셋
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        # 평균 처리 시간 텍스트 영상에 출력
        if hasattr(self, "last_avg_text"):
            cv2.putText(cv_image, self.last_avg_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        #---------------------------------------------------------------------------------------------------- 결과
        
        cv2.imshow("Mask", hsv_mask)    # 마스크 흑백
        cv2.imshow("Subscriber", cv_image)  # 결과
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()