import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2


class ImgSubscriber(Node):
    def __init__(self):
        super().__init__('img_subscriber')
        self.subscription_color = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # 컬러 이미지 토픽
            self.color_image_callback, 10)        
        self.bridge = CvBridge()

    def color_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("image", cv_image)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = ImgSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()