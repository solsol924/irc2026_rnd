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
from robot_msgs.msg import LinePointsArray, LineResult, MotionEnd # type: ignore
from rcl_interfaces.msg import SetParametersResult

camera_width = 640
camera_height = 480

roi_x_start = int(camera_width * 0 // 5)
roi_x_end = int(camera_width * 5 // 5)
roi_y_start = int(camera_height * 1 // 12)
roi_y_end = int(camera_height * 11 // 12)

zandi_x = int((roi_x_start + roi_x_end) / 2)
zandi_y = int((roi_y_start + roi_y_end) / 2) + 140

class LineListenerNode(Node): #################################################################### 판단 프레임 수 바꿀 때 yolo_cpp 도 고려해라~~~
    def __init__(self):
        super().__init__('line_subscriber')

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)

        self.last_avg_text = "AVG: --- ms | FPS: --"        

        
        self.collecting = False     # 수집 중 여부
        self.armed = False               # motion_end 방어~!

        self.window_id = 0
        self.frame_idx = 0      
        self.frames_left = 0       # 남은 프레임 수 < collecting_frames
        
        self.last_angle = 0
        self.last_line_angle = 0.0
        self.last_delta_zandi = 0.0
        self.last_status = 0
        self.last_line_xy = None    

        self.status_list = [] # 누적 값 저장
        self.angle_list = []
        self.avg_x_list = []

        self.miss_count = 0
        self.out_count = 0

        self.bridge = CvBridge()
        self.sub = self.create_subscription(  # 중심점 토픽
            LinePointsArray,                   
            'candidates',                   
            self.line_callback, 10)      
        self.motion_end_sub = self.create_subscription(  # 모션 끝나면 받아올 T/F
            MotionEnd,                   
            '/motion_end',                   
            self.motion_callback, 10)                         
        self.subscription_color = self.create_subscription(  # 이미지 토픽
            Image,
            '/cam1/color/image_raw',  #  640x480 / 15fps cam2
            self.color_image_callback, 10)
        
        # 파라미터 선언
        self.declare_parameter("delta_zandi_min", 190) # 핑크 선 두께
        self.declare_parameter("collecting_frames", 9) # 몇 프레임 동안 보고 판단할 거냐  << yolo는 15프레임, 정지 5프레임, 탐지는 10프레임만
        self.declare_parameter("delta_tip_min", 15) # 빨간 점이 얼마나 벗어나야 커브일까
        self.declare_parameter("vertical", 15) # 직진 판단 각도
        self.declare_parameter("horizontal", 75) # 수평 판단 각도  <<<  아직 안 만듬
        self.declare_parameter("delta_out", 120) # out 복귀 판단 범위

        # 파라미터 적용
        self.delta_zandi_min = self.get_parameter("delta_zandi_min").value
        self.collecting_frames = self.get_parameter("collecting_frames").value
        self.delta_tip_min = self.get_parameter("delta_tip_min").value
        self.vertical = self.get_parameter("vertical").value
        self.horizontal = self.get_parameter("horizontal").value
        self.delta_out = self.get_parameter("delta_out").value
        
        self.candidates = [] 
        self.curve_count = 0
        self.tilt_text = "" # 화면 출력
        self.curve_text = "" # 화면 출력
        self.out_text = "" # 화면 출력

        self.out_now = False
        self.out_angle = 0
        self.last_mean_angle = 0
        self.last_avg_x = (roi_x_start + roi_x_end) / 2

        self.add_on_set_parameters_callback(self.param_callback)

        # 퍼블리셔
        self.line_result_pub = self.create_publisher(LineResult, '/line_result', 10)

    def param_callback(self, params):
        for p in params:
            if p.name == "delta_zandi_min": self.delta_zandi_min = int(p.value)
            elif p.name == "collecting_frames": self.collecting_frames = int(p.value)
            elif p.name == "delta_tip_min": self.delta_tip_min = int(p.value)
            elif p.name == "vertical": self.vertical = int(p.value)
            elif p.name == "horizontal": self.horizontal = int(p.value)
            elif p.name == "delta_out": self.delta_out = int(p.value)
            
        return SetParametersResult(successful=True)

    def motion_callback(self, msg: MotionEnd): # 모션 끝 같이 받아오기 (중복 방지)
        if bool(msg.motion_end_detect):
            self.armed = True
            self.get_logger().info("Subscribed /motion_end !!!!!!!!!!!!!!!!!!!!!!!!")

    def line_callback(self, msg: LinePointsArray): # yolo_cpp에서 토픽 보내면 실핼
        new_candidates = [(i.cx, i.cy, i.lost) for i in msg.points]  # 먼저 로컬 변수에만
        if not self.collecting:
            if not self.armed:
                return
            
            self.collecting = True
            self.armed = False
            self.frames_left = self.collecting_frames
            self.window_id += 1
            
            # 초기화
            self.status_list.clear()
            self.angle_list.clear()
            self.avg_x_list.clear()
            self.frame_idx = 0

            self.line_start_time = time.time()

            self.get_logger().info(f'[Start] Window {self.window_id} | I got {self.collecting_frames} frames')

        # 진짜
        self.candidates = new_candidates
        self.frame_idx += 1
        self.get_logger().info(f"step {self.frame_idx}")
        for idx, (cx, cy, lost) in enumerate(self.candidates):
            self.get_logger().info(f'[{idx}] cx={cx}, cy={cy}, lost={lost}')

        line_angle = 0.0 # 직선 각도
        angle = 0.0 # 잔디가 회전할 최종 각도
        delta_zandi = 0
        status = 0

        if len(self.candidates) >= 3:  # =================================================== I. 세 점 이상 탐지
            tip_x, tip_y, tip_lost = self.candidates[-1] # 맨 위의 점
            line_x = np.array([c[0] for c in self.candidates[:-1]], dtype=np.float32) # 나머지 점
            line_y = np.array([c[1] for c in self.candidates[:-1]], dtype=np.float32)
            avg_line_x = int(round(np.mean(line_x)))
            self.last_avg_x = avg_line_x

            if float(line_x.max() - line_x.min()) < 2.0: # 거의 일직선이면 (계산식 발산 방지)
                x1 = x2 = avg_line_x
                y1 = roi_y_end; y2 = roi_y_start
                line_angle = 0 # tilt
                delta_tip = float(avg_line_x - tip_x) # 1. curve
                delta_zandi = float(avg_line_x - zandi_x) # 3. out
            else: # 일반적인 경우
                pts = np.stack([line_x, line_y], axis=1)
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)
                m = vy / vx

                line_angle = math.degrees(math.atan2(vx, -vy))
                if line_angle >= 90:
                    line_angle -= 180

                signed_tip = (tip_x - x0) * vy - (tip_y - y0) * vx
                signed_zandi = (zandi_x - x0) * vy - (zandi_y - y0) * vx
                delta_tip = signed_tip / (math.hypot(vx, vy) + 1e-5)
                delta_zandi = signed_zandi / (math.hypot(vx, vy) + 1e-5)
                delta_zandi = delta_zandi * abs(line_angle) / line_angle 

                b = y0 - m * x0
                if abs(m) > 1:
                    y1, y2 = roi_y_end, roi_y_start
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:
                    x1, x2 = roi_x_end, roi_x_start
                    y1 = int(m * x1 + b)
                    y2 = int(m * x2 + b)

            # ------------------------------------------------------------------------------------------- 판단 시작
            if self.out_now: # 한 번 나간 상태
                if abs(delta_zandi) <= self.delta_out: # 탈출 성공
                    if abs(line_angle) < self.vertical:
                        angle = 0
                        status = 1
                        self.tilt_text = "Straight"
                    elif self.horizontal > line_angle > self.vertical:
                        angle = line_angle
                        status = 3
                        self.tilt_text = "Spin Right"
                    elif -self.horizontal < line_angle < -self.vertical:
                        angle = line_angle
                        status = 2
                        self.tilt_text = "Spin Left"
                else:
                    angle = 0
                    status = 12  # 탈출 실패 
                    self.out_text = "Still Out"

            else: # 평소
                if abs(delta_tip) > self.delta_tip_min: # 분기 1-1. Turn = RL
                    angle = math.degrees(math.atan2(tip_x - zandi_x, -(tip_y - zandi_y)))
                    if angle >= 90:
                        angle -= 180
                    if abs(angle) <= self.vertical: 
                        status = 1
                    elif angle > self.vertical: 
                        status = 3
                    elif angle < -self.vertical:
                        status = 2

                    self.curve_text = "Turn Left" if (delta_tip * line_angle > 0) else "Turn Right"

                    if abs(delta_zandi) <= self.delta_zandi_min: # 분기 3-1 Out = In
                        self.out_text = "In"
                        if abs(line_angle) < self.vertical:
                            self.tilt_text = "Straight"
                        elif self.horizontal > line_angle > self.vertical:
                            self.tilt_text = "Spin Right"
                        elif -self.horizontal < line_angle < -self.vertical:
                            self.tilt_text = "Spin Left"
                    else: # 분기 3-2 Out = RL
                        self.out_text = "Out Left" if (delta_zandi > 0) else "Out Right"
                        if abs(line_angle) < self.vertical:
                            self.tilt_text = "Straight"
                        elif self.horizontal > line_angle > self.vertical:
                            self.tilt_text = "Spin Right"
                        elif -self.horizontal < line_angle < -self.vertical:
                            self.tilt_text = "Spin Left"
                else: # 분기 1-2 Turn = Straight
                    self.curve_text = "Straight"
                    if abs(delta_zandi) < self.delta_zandi_min: # 분기 3-1 Out = In
                        self.out_text = "In"
                        if abs(line_angle) < self.vertical: # 분기 2 Tilt
                            angle = 0
                            status = 1
                            self.tilt_text = "Straight"
                        elif self.horizontal > line_angle > self.vertical:
                            angle = line_angle
                            status = 3
                            self.tilt_text = "Spin Right"
                        elif -self.horizontal < line_angle < -self.vertical:
                            angle = line_angle
                            status = 2
                            self.tilt_text = "Spin Left"
                    else: # 분기 3-2 Out = RL >> 복귀 모션 ㄱㄱ  >>>> 여기 위에랑 뭐가 다르냐
                        self.out_text = "Out Left" if (delta_zandi > 0) else "Out Right"

                        r = float(np.clip(delta_zandi / 320.0, -1.0, 1.0))
                        angle = math.degrees(math.asin(r)) + line_angle

                        if abs(line_angle) < self.vertical:
                            self.tilt_text = "Straight"
                        elif self.horizontal > line_angle > self.vertical:
                            self.tilt_text = "Spin Right"
                        elif -self.horizontal < line_angle < -self.vertical:
                            self.tilt_text = "Spin Left"

                        if abs(angle) < self.vertical:
                            status = 12 # 한 발자국
                        elif self.horizontal > angle > self.vertical:
                            status = 14 # 우회전 후 한 발자국
                        elif -self.horizontal < angle < -self.vertical:
                            status = 13 # 좌회전 후 한 발자국

            self.last_line_xy = (int(x1), int(y1), int(x2), int(y2)) # 시각화

        elif len(self.candidates) == 2:  # ==================================================== II. 2점 탐지
            self.curve_text = "Straight"

            down_x, down_y, down_lost = self.candidates[0]
            up_x, up_y, up_lost = self.candidates[1]
            avg_line_x = int(round((down_x + up_x)/2))

            if abs(down_x - up_x) < 2: # 일직선인 경우
                line_angle = 0
                x1 = x2 = avg_line_x
                y1, y2 = roi_y_end, roi_y_start
                delta_zandi = abs(float(avg_line_x - zandi_x))
            else: # 일반적인 경우
                m = (up_y - down_y) / (up_x - down_x)
                b = down_y - m * down_x
                line_angle = math.degrees(math.atan2(up_x - down_x, up_y - down_y))

                if line_angle > 90:
                    line_angle -= 180
                signed_zandi = (zandi_x - down_x)*(up_y - down_y) - (zandi_y - down_y)*(up_x - down_x)
                delta_zandi = abs(signed_zandi / (math.hypot(up_x - down_x, up_y - down_y) + 1e-5))
                
                if abs(m) > 1:
                    y1, y2 = roi_y_end, roi_y_start
                    x1 = int((y1 - b)/m); x2 = int((y2 - b)/m)
                else:
                    x1, x2 = roi_x_end, roi_x_start
                    y1 = int(m*x1 + b); y2 = int(m*x2 + b)

            if delta_zandi < self.delta_zandi_min: # 분기 3-1 Out = In
                self.out_text = "In"

                if abs(line_angle) <= self.vertical: # 분기 2 Tilt
                    angle = 0
                    status = 1
                    self.tilt_text = "Straight"
                elif self.horizontal > line_angle > self.vertical:
                    angle = line_angle
                    status = 3
                    self.tilt_text = "Spin Right"
                elif -self.horizontal < line_angle < -self.vertical:
                    angle = line_angle
                    status = 2
                    self.tilt_text = "Spin Left"

            else: #분기 3-2 Out = RL  >> 복귀 모션 ㄱㄱ~
                self.out_text = "Out Left" if (avg_line_x - zandi_x > 0) else "Out Right"
                r = float(np.clip(delta_zandi / 320.0, -1.0, 1.0))
                angle = math.degrees(math.asin(r))

                if abs(line_angle) <= self.vertical: # 분기 2 Tilt
                    self.tilt_text = "Straight"
                elif self.horizontal > line_angle > self.vertical:
                    self.tilt_text = "Spin Right"
                elif -self.horizontal < line_angle < -self.vertical:
                    self.tilt_text = "Spin Left"
                angle += line_angle

                if abs(angle) <= self.vertical: 
                    status = 12 # 한 발자국
                elif self.horizontal > angle > self.vertical: 
                    status = 14 # 우회전 후 ㅎ ㅂㅈㄱ
                elif -self.horizontal < angle < -self.vertical:
                    status = 13 # 좌회전 후 ㅎ ㅂㅈㄱ

            self.last_line_xy = (int(x1), int(y1), int(x2), int(y2))

        else:  # ======================================================================================= III. 1점 이하 탐지
            self.curve_text = "Miss"
            self.tilt_text = "Miss"
            self.out_text = "Miss"

            angle = 0
            line_angle = 0
            delta_zandi = 0
            status = 5
            avg_line_x = int(round((roi_x_start + roi_x_end) / 2))

            self.last_line_xy = None
        # =========================================================================================================

        # 결과 누적/업데이트
        angle = int(round(angle))
        self.frames_left -= 1
        self.status_list.append(status)
        self.angle_list.append(angle)
        self.avg_x_list.append(avg_line_x)

        # 화면용 최신 값 저장
        self.last_angle = angle
        self.last_line_angle = int(round(line_angle))
        self.last_delta_zandi = float(round(delta_zandi,1))
        self.last_status = status

        # 15프레임 다 끝나면
        if self.frames_left <= 0:
            cnt = Counter(self.status_list)
            res = max(cnt.items(), key=lambda kv: kv[1])[0]
            angles = [a for s, a in zip(self.status_list, self.angle_list) if s == res]
            mean_angle = int(round(np.mean(angles)))
            self.last_mean_angle = mean_angle
            mean_avg_x = int(round(np.mean([a for s, a in zip(self.status_list, self.avg_x_list) if s == res]))) if res != 5 else self.last_avg_x # 계속 예전 값 유지
            self.last_avg_x = mean_avg_x

            if self.out_now:
                if res != 12: # 탈출 성공
                    self.out_now = False
                    self.out_count = 0
                    self.get_logger().info(f"[Line] In! | res= {res}, angle_avg= {mean_angle}")
                else: # 탈출 실패
                    self.out_count += 1
                    if self.out_count >= 5: # 5연속 탈출 못 했으면 때려쳐라
                        self.out_now = False    
                        self.out_count = 0
                        res = 5
                        self.get_logger().info(f"[Line] Still out,, | res= {res}, angle_avg= {mean_angle}")
                    else:
                        self.get_logger().info(f"[Line] Recovering! | res= {res}, angle_avg= {mean_angle}, out_count= {self.out_count}")
            else: # 평소
                if res in (12, 13, 14): # out이 되었다면
                    self.out_now = True
                    self.out_count += 1
                    self.get_logger().info(f"[Line] Out! | res= {res}, angle_avg= {mean_angle}")

            if res != 5:
                self.miss_count = 0
                self.get_logger().info(f"[Line] Found | res= {res}, angle_avg= {mean_angle}, line_angle= {self.last_line_angle}, " # line_angle은 임시
                                    f"delta= {self.last_delta_zandi}")
            else:
                self.miss_count += 1
                if self.miss_count >= 5:
                    self.miss_count = 5 
                    res = 5 # 5번 연속으로 점선을 못 찾는다면?
                    self.get_logger().info(f"[Line] I totally missed,,, | res= {res}, miss over {self.miss_count} times")
                elif self.miss_count >= 2:
                    if self.last_avg_x > (roi_x_start + roi_x_end) / 2 + self.delta_zandi_min:
                        res = 3
                        mean_angle = 45
                        self.get_logger().info(f"[Line] Miss,, maybe right | res= {res}, miss count= {self.miss_count}") # 못 찾았고 이전 평균 라인 x좌표가 오른쪽
                    elif self.last_avg_x < (roi_x_start + roi_x_end) / 2 - self.delta_zandi_min:
                        res = 2
                        mean_angle = -45
                        self.get_logger().info(f"[Line] Miss,, maybe left | res= {res}, miss count= {self.miss_count}") # 못 찾았고 이전 평균 라인 x좌표가 왼쪽
                    else:
                        res = 5
                        self.get_logger().info(f"[Line] Miss | res= {res}, miss count= {self.miss_count}") # 그것도 아닌데 못 찾았으면 일단 뒤로가기
                else:
                    res = 5
                    self.get_logger().info(f"[Line] Miss | res= {res}, miss count= {self.miss_count}") # 한 번 못 찾았으면 일단 뒤로가기

            process_time = (time.time() - self.line_start_time) / self.collecting_frames if self.line_start_time is not None else 0.0

            # 퍼블리시
            msg_out = LineResult()
            msg_out.res = res
            msg_out.angle = abs(mean_angle)
            self.line_result_pub.publish(msg_out)
            self.get_logger().info(f'frames= {len(self.status_list)}, wall= {process_time*1000:.1f} ms')

            # 리셋
            self.collecting = False
            self.frames_left = 0
            self.candidates = []
            self.last_line_xy = None     # ★ 그래프 클리어
            self.frame_idx = 0

            self.out_text = ""
            self.curve_text = ""
            self.tilt_text = ""

    def color_image_callback(self, msg): # 단순 출력용 >> 실제 쓸 땐 다 주석처리~
        start_time = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        step_text = f"{self.frame_idx}/{self.collecting_frames}"

        if self.candidates is not None:
            # 후보 점 시각화 (최근 candidates 기준)
            for i, (cx, cy, lost) in enumerate(self.candidates):
                if lost > 0:
                    color = (0, 255, 255)
                else:
                    if i == len(self.candidates) - 1:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)
                cv2.circle(roi, (cx - roi_x_start, cy - roi_y_start), 3, color, -1)

            # 마지막 계산된 직선
            if self.last_line_xy is not None:
                x1, y1, x2, y2 = self.last_line_xy
                cv2.line(roi, (int(x1 - roi_x_start), int(y1 - roi_y_start)),
                            (int(x2 - roi_x_start), int(y2 - roi_y_start)), (255, 0, 0), 2)

            cv2.putText(cv_image, f"Step: {step_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # HUD 텍스트 (마지막 계산 결과)
        cv2.putText(cv_image, f"Rotate: {self.last_mean_angle}", (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(cv_image, f"Tilt: {self.tilt_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        cv2.putText(cv_image, f"Curve: {self.curve_text}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        cv2.putText(cv_image, f"Out: {self.out_text}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        # ROI/잔디 기준 시각화
        cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = roi
        cv2.rectangle(cv_image, (roi_x_start + 1, roi_y_start + 1), (roi_x_end - 1, roi_y_end - 1), (0, 255, 0), 1)
        cv2.circle(cv_image, (zandi_x, zandi_y), 5, (111,255,111), -1)
        cv2.circle(cv_image, (zandi_x, zandi_y), self.delta_zandi_min, (255,22,255), 1)
        cv2.circle(cv_image, (zandi_x, zandi_y), self.delta_out, (255,22, 22), 1)

        # PING/FPS
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        if time.time() - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            avg_fps = self.frame_count / (time.time() - self.last_report_time)
            self.last_avg_text = f"PING: {avg_time*1000:.2f}ms | FPS: {avg_fps:.2f}"
            self.frame_count = 0; self.total_time = 0.0; self.last_report_time = time.time()
        cv2.putText(cv_image, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Line", cv_image)
        cv2.waitKey(1)

        

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()