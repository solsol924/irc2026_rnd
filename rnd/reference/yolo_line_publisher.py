import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import cv2
import time
from my_cv_msgs.msg import LinePoint, LinePointsArray, MotionEnd # type: ignore 

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

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')     
        self.subscription = self.create_subscription( # 화면 받아오고
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.line_publisher = self.create_publisher(LinePointsArray, 'candidates', 10) # 욜로 결과 퍼블리시

        self.model = torch.hub.load( # 욜로 모델
            'ultralytics/yolov5',
            'custom',
            path='/home/noh/my_cv/ultralytics/yolov5/runs/train/line_yolo/weights/best.pt',
            source='local'
        )

        self.model.conf = 0.5  # 신뢰도 0
        self.bridge = CvBridge()

        self.tracker = RectangleTracker(10, 60, 10)

        self.frame_count = 0
        self.total_time = 0.0
        self.last_detected_time = time.time()
        self.last_report_time = time.time()
        self.yolo = False
        self.last_avg_text = "AVG: --- ms | FPS: --"

        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL) # 창 조절 가능

    def image_callback(self, msg):

        start_time = time.time()  # 시작 시간 기록
        
        bgr_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # CV는 색공간을 BGR로 받아옴
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # 욜로는 RBG로 보기 때문에 여기서 바꿔준다 (우리도)
        results = self.model(rgb_frame) # 그래서 욜로 모델도 RGB

        screen_h, screen_w, _ = bgr_frame.shape
        roi_y_start = screen_h * 1 // 5  # 위
        roi_y_end = screen_h // 1        # 아래
        roi_x_start = screen_w * 1 // 5  # 왼 
        roi_x_end = screen_w * 4 // 5    # 오
        
        detections = results.pandas().xyxy[0]
        centers = []

        if not detections.empty:
            for _, row in detections.iterrows():
                label = row['name']
                conf = row['confidence']
                if label == 'line': # 욜로 클래스 중 line만 처리
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if roi_x_end >= cx >= roi_x_start and roi_y_end >= cy >= roi_y_start:
                        centers.append((cx, cy)) # 중심점 배열에 추가

                    cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(bgr_frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(bgr_frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            valid_rects = self.tracker.update(centers)
            candidates = sorted(
                        [(cx, cy, lost) for (cx, cy, lost, found) in valid_rects.values()],
                        key=lambda c: c[1], reverse=True)[:5] # 후보점 5개 선택 (화면 아래쪽부터)
        else:
            candidates = []
        
        msg_array = LinePointsArray() # 튜플로
        for (cx, cy, lost) in candidates: 
            msg = LinePoint() # (cx, cy, lost)
            msg.cx = cx
            msg.cy = cy
            msg.lost = lost
            msg_array.points.append(msg) # [(cx, cy, lost), (cx, cy, lost),,,,,,,,,]
        self.line_publisher.publish(msg_array)
        print(msg_array)

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
            cv2.putText(bgr_frame, self.last_avg_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Detection", bgr_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
