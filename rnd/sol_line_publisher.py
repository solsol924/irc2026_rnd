import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# ROS 2 표준 메시지 임포트
from geometry_msgs.msg import Polygon, Point32

class LinePublisher(Node):
    def __init__(self):
        super().__init__('sol_line_publisher')
        # 카메라 영상 구독 (토픽 이름은 환경에 맞게 수정하세요)
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        # 중심점 리스트를 발행할 퍼블리셔 (Polygon 타입)
        self.publisher = self.create_publisher(Polygon, 'track_centers', 10)
        
        self.bridge = CvBridge()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def image_callback(self, msg):
        # 1. ROS 이미지를 OpenCV로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 2. 전처리 (이진화)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        dilation = cv2.dilate(thresh, self.kernel)
        
        # 3. 윤곽선 찾기
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 모든 점을 담을 Polygon 메시지 생성
        msg_polygon = Polygon()
        
        for cnt in contours:
            # 면적 필터링 (너무 작은 노이즈 제거)
            if cv2.contourArea(cnt) > 500:
                # 최소 외접 사각형 계산
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                
                # 4. 각 사각형의 중심점을 Point32에 담아 리스트에 추가
                p = Point32()
                p.x = float(int(x)) # 중심점 x
                p.y = float(int(y)) # 중심점 y
                p.z = 0.0           # 사용 안 함 (필요시 lost 값 등으로 활용 가능)
                
                msg_polygon.points.append(p)
                
                # 화면 표시용 (빨간 점)
                cv2.circle(cv_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 5. 최종 발행
        self.publisher.publish(msg_polygon)
        
        # 로그 출력
        self.get_logger().info(f'Published {len(msg_polygon.points)} center points.')
        
        # 결과 화면 보기
        cv2.imshow("Solhee Publisher", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LinePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()