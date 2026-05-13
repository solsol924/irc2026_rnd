import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# YOLOv8 불러오기
from ultralytics import YOLO

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # ROS 2 이미지 변환기
        self.bridge = CvBridge()
        
        # YOLOv8 Nano 모델 로드 (가장 가볍고 빠른 모델)
        # 처음 실행할 때 yolov8n.pt 파일을 자동으로 다운로드합니다.
        self.model = YOLO('yolov8n.pt') 
        
        # 카메라 이미지 구독 (웹캠이나 RealSense에서 퍼블리시하는 토픽 이름)
        # 상황에 맞게 '/camera/image_raw' 부분을 수정하세요.
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info("YOLOv8 노드가 시작되었습니다! 카메라 영상을 기다립니다...")

    def image_callback(self, msg):
        # 1. ROS 2 이미지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 2. YOLOv8로 객체 인식 실행
        results = self.model(cv_image, verbose=False)
        
        # 3. 인식된 결과(네모 박스)가 그려진 이미지 가져오기
        annotated_frame = results[0].plot()
        
        # 4. 화면에 출력
        cv2.imshow("YOLOv8 ROS2 Detection", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()