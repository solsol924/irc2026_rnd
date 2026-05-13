from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# 메세지
from message_filters import Subscriber, ApproximateTimeSynchronizer

import rclpy as rp
import numpy as np
import cv2

class HurdleDetectorNode(Node):
    def __init__(self):
        super().__init__('hurdle_detector')

        # 변수
        self.lower_hsv = np.array([20, 20, 70])
        self.upper_hsv = np.array([80, 255, 255]) # 노랑색 기준으로
        self.depth_thresh = 500.0  # mm 기준 마스크 거리

        # 컬러, 깊이 영상 동기화
        self.bridge = CvBridge()
        color_sub = Subscriber(self, Image, '/camera/color/image_raw') # 칼라
        depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw') # 깊이
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.image_callback)

    def image_callback(self, color_msg: Image, depth_msg: Image):

        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        frame[depth >= self.depth_thresh] = 0 # 거리
        frame[depth <= 30.0] = 0

        cv2.imshow('Raw Hurdle Mask', frame) # 기준 거리 이내
        cv2.waitKey(1)

def main():
    rp.init()
    node = HurdleDetectorNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()
