#!/usr/bin/env python3
import rclpy
import numpy as np
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rcl_interfaces.msg import SetParametersResult

class ColorNode(Node):
    def __init__(self):
        super().__init__('color_test_node')

        self.bridge = CvBridge()
        self.color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        # 파라미터 선언
        self.declare_parameter("h_low", 8) # 색상
        self.declare_parameter("h_high", 60)
        self.declare_parameter("s_low", 60) # 채도
        self.declare_parameter("s_high", 255)
        self.declare_parameter("v_low", 0) # 밝기
        self.declare_parameter("v_high", 255)
        self.declare_parameter("depth_min", 50)
        self.declare_parameter("depth_max", 1000)

        # 파라미터 적용
        self.h_low = self.get_parameter("h_low").value
        self.h_high = self.get_parameter("h_high").value
        self.s_low = self.get_parameter("s_low").value
        self.s_high = self.get_parameter("s_high").value
        self.v_low = self.get_parameter("v_low").value
        self.v_high = self.get_parameter("v_high").value
        self.depth_min = self.get_parameter("depth_min").value
        self.depth_max = self.get_parameter("depth_max").value

        self.add_on_set_parameters_callback(self.param_callback)

        self.lower_hsv = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        self.upper_hsv = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
        self.hsv = None

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 화면 클릭
        cv2.namedWindow('Color Mask')
        cv2.setMouseCallback('Color Mask', self.click)

    def param_callback(self, params):
        for p in params:
            if p.name == "h_low": self.h_low = int(p.value)
            elif p.name == "h_high": self.h_high = int(p.value)
            elif p.name == "s_low": self.s_low = int(p.value)
            elif p.name == "s_high": self.s_high = int(p.value)
            elif p.name == "v_low": self.v_low = int(p.value)
            elif p.name == "v_high": self.v_high = int(p.value)
            elif p.name == "depth_min": self.depth_min = int(p.value)
            elif p.name == "depth_max": self.depth_max = int(p.value)

        # HSV 갱신
        self.lower_hsv = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        self.upper_hsv = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)

        return SetParametersResult(successful=True)

    def click(self, event, x, y, _, __):  # 화면 클릭
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        H, S, V = [int(v) for v in self.hsv[y, x]]
        self.get_logger().info(f"[Pos] x={x}, y={y} | HSV=({H},{S},{V})")

    def image_callback(self, color_msg: Image, depth_msg: Image):
        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        # HSV 색 조절
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(self.hsv, self.lower_hsv, self.upper_hsv)
        mask = raw_mask.copy()

        # 깊이 마스크 생성
        depth_mask = np.zeros_like(depth, dtype=np.uint8)
        depth_mask[(depth > self.depth_min) & (depth < self.depth_max)] = 255

        # 모폴로지 연산
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, self.kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, self.kernel)

        # 마스크 흑백 > 컬러    <<< 합치려고
        raw_mask_color = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 깊이 마스크
        depth_mask_bool = (depth > self.depth_min) & (depth < self.depth_max)
        color_in_depth = np.zeros_like(frame)
        color_in_depth[depth_mask_bool] = frame[depth_mask_bool]

        # 컬러 3분할
        concat_img = np.hstack([raw_mask_color, mask_color, frame])
        cv2.imshow('Color Mask', concat_img)

        # 거리 2분할
        depth_mask_bgr = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
        depth_concat = np.hstack([depth_mask_bgr, color_in_depth])
        cv2.imshow('Depth Mask', depth_concat)

        cv2.waitKey(1)

def main():
    rclpy.init()
    node = ColorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
