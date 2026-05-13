
# CLAHE 대신 평활화 사용 -> 대략 1ms 더 빠름

import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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


class ImgSubscriber(Node):
    def __init__(self):
        super().__init__('line_publisher')
        self.subscription_color = self.create_subscription(  # 컬러
            Image,
            '/camera/color/image_raw',  # RealSense에서 제공하는 컬러 이미지 토픽  >>  640x480 / 15fps
            self.color_image_callback, 10)
        
        self.line_publisher = self.create_publisher(
            LinePointsArray, 
            'candidates', 
            10) # 중심점 퍼블리싱

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
                
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # 시간 디버깅
        self.frame_count = 0
        self.total_time = 0.0
        self.last_detected_time = time.time()
        self.last_report_time = time.time()
        self.yolo = False
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

    def color_image_callback(self, msg):

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

            # HSV 분리 후 V 채널 평활화 (equalizeHist)
            h, s, v = cv2.cuda.split(gpu_hsv)
            v = cv2.cuda.equalizeHist(v)

            # HSV 재조합
            gpu_hsv_eq = cv2.cuda_GpuMat(gpu_hsv.size(), gpu_hsv.type())
            cv2.cuda.merge([h,s,v], gpu_hsv_eq)

            # 2. HSV 흰색 범위 지정
            hsv_upper_white = (self.h_up, self.s_up, self.v_up)
            hsv_lower_white = (self.h_down, self.s_down, self.v_down)
            hsv_mask = cv2.cuda.inRange(gpu_hsv_eq, hsv_lower_white, hsv_upper_white) # 이 화면으로 판단

            # GPU 메모리에서 HSV 이미지 다운로드   ---------------------------------------------------------------------------------------CUDA 끝
            hsv_image = hsv_mask.download()   
            
            #--------------------------------------------------------------------------------------------------------------------- 보정 끝

            # BGR 이미지로 변환
            #bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # 보정 끝낸 bgr 화면 < 내가 보려고 띄우는 화면

            contours, _ = cv2.findContours(hsv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                self.last_detected_time = time.time()
                self.yolo = False
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
                                approx_shifted = approx + np.array([[[roi_x_start, roi_y_start]]]) # 평행이동
                                cv2.drawContours(cv_image, [approx_shifted], 0, (200, 0, 255), 2)  # 테두리 핑크 / 사각형 검출 하는지 확인용 / 지금 프레임에서 사각형이라고 검출된 모든 사각형
                                rect_centroids.append((box_cx,box_cy))  # 사각형 중심 리스트에 추가 
                                    
                # -------------------------------------------------------------------- 사각형 추적

                valid_rects = self.tracker.update(rect_centroids)  # 사각형 중심을 추적기에 보냄 {id: (cx, cy, lost, found)} / lost & found 조건도 만족한 사각형            
                for id,(cx,cy,lost,found) in valid_rects.items(): 
                    cv2.putText(cv_image, f"ID: {id}", (cx + roi_x_start, cy + roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # 유효 사각형은 id 출력

                #---------------------------------------------------------------------- 라인 판단할 점들 발행
                
                candidates = sorted(
                    [(cx, cy, lost) for (cx, cy, lost, found) in valid_rects.values()],
                    key=lambda c: c[1], reverse=True)[:5] # 후보점 5개 선택 (화면 아래쪽부터)
            else:
                candidates = []
                if time.time() - self.last_detected_time >= 10:
                    self.yolo = True
                    print("Miss")
            
            msg_array = LinePointsArray() # 튜플로
            for (cx, cy, lost) in candidates: 
                msg = LinePoint() # (cx, cy, lost)
                msg.cx = cx
                msg.cy = cy
                msg.lost = lost
                msg_array.points.append(msg) # [(cx, cy, lost), (cx, cy, lost),,,,,,,,,]
            self.line_publisher.publish(msg_array)
            print(msg_array)

            #cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = bgr_image  # 전체화면
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

            cv2.imshow("Mask", hsv_image)    # 마스크 흑백
            cv2.imshow("Publisher2", cv_image)  # 결과
            cv2.waitKey(1)

def main():
    rp.init()
    node = ImgSubscriber()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()