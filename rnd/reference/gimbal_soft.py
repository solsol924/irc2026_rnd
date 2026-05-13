#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge

import cv2
import numpy as np


class GimbalSoftNode(Node):
    def __init__(self):
        super().__init__("gimbal_soft")

        # ===========================
        # 파라미터 선언
        # ===========================
        self.declare_parameter("crop_ratio", 0.85)      # 중앙에서 얼마나 크롭할지 (0~1)
        self.declare_parameter("follow_alpha", 0.98)    # EMA 비율 (1에 가까울수록 느리게, 더 안정적)
        self.declare_parameter("fx", 607.0)             # 카메라 내참값 (대략)
        self.declare_parameter("fy", 606.0)
        self.declare_parameter("roll_gain", 1.0)        # roll 보정 강도
        self.declare_parameter("pitch_gain", 1.0)       # pitch 보정 강도
        self.declare_parameter("yaw_gain", 1.0)         # yaw 보정 강도
        self.declare_parameter("zoom", 1.05)            # 회전 시 확대 비율(마진 확보용)

        # 파라미터 로드
        self.crop_ratio = float(self.get_parameter("crop_ratio").value)
        self.follow_alpha = float(self.get_parameter("follow_alpha").value)
        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.roll_gain = float(self.get_parameter("roll_gain").value)
        self.pitch_gain = float(self.get_parameter("pitch_gain").value)
        self.yaw_gain = float(self.get_parameter("yaw_gain").value)
        self.zoom = float(self.get_parameter("zoom").value)

        # ===========================
        # IMU 상태
        #  - roll, pitch, yaw : 실제 IMU에서 추정된 값 (rad)
        #  - *_smooth         : EMA로 부드럽게 만든 "가상 카메라" 자세
        # ===========================
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.roll_smooth = 0.0
        self.pitch_smooth = 0.0
        self.yaw_smooth = 0.0

        self.have_imu = False

        self.bridge = CvBridge()

        # ===========================
        # 구독자
        # ===========================
        self.sub_img = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            10
        )

        # /camera/imu_tilt : Mahony / ImuTiltNode에서 나온 (roll, pitch, yaw)
        # 여기선 카메라 설치 방향에 맞게 축을 한번 더 매핑
        self.sub_imu = self.create_subscription(
            Vector3Stamped,
            "/camera/imu_tilt",
            self.imu_callback,
            50
        )

        self.get_logger().info("Soft Gimbal Viewer started.")

        # OpenCV 윈도우 이름
        self.window_name = "Soft Gimbal"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    # ==========================================
    # IMU Callback
    #  - IMU에서 받은 (roll, pitch, yaw)를
    #    카메라 영상 기준 좌표계에 맞게 매핑
    #  - EMA(지수 이동 평균)으로 가상 카메라 자세 생성
    # ==========================================
    def imu_callback(self, msg: Vector3Stamped):
        # 예시 매핑 (필요에 따라 조정하신 내용 유지)
        #  - msg.vector.x : IMU 기준 roll
        #  - msg.vector.y : IMU 기준 pitch
        #  - msg.vector.z : IMU 기준 yaw
        # 카메라 장착 각도에 따라 다음과 같이 뒤집어 쓰는 중:
        self.roll = msg.vector.y       # 카메라 기준 roll
        self.pitch = -msg.vector.x     # 카메라 기준 pitch (부호 반전)
        self.yaw = msg.vector.z        # 카메라 기준 yaw

        self.have_imu = True

        # EMA filtering of "virtual camera" (느리게 따라가는 기준자세)
        a = self.follow_alpha
        self.roll_smooth = a * self.roll_smooth + (1.0 - a) * self.roll
        self.pitch_smooth = a * self.pitch_smooth + (1.0 - a) * self.pitch
        self.yaw_smooth = a * self.yaw_smooth + (1.0 - a) * self.yaw

    # ==========================================
    # Stabilization Function
    #  - 현재 자세 vs 부드러운 자세 차이(고주파 흔들림)를
    #    roll: 회전, pitch/yaw: 평행이동으로 보정
    # ==========================================
    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.have_imu:
            # IMU 아직 없으면 원본 그대로
            return frame

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # 1) difference = high-frequency shake (현재 자세 - EMA)
        roll_err = self.roll - self.roll_smooth
        pitch_err = self.pitch - self.pitch_smooth
        yaw_err = self.yaw - self.yaw_smooth

        # 2) roll_err → 영상 회전 (광학축 주변 회전)
        angle_deg = - roll_err * 180.0 / math.pi * self.roll_gain

        # 3) yaw_err, pitch_err → 영상 평행이동
        #    소각 근사 : 이미지 평면에서
        #      yaw(수평 회전)   → x 방향 이동
        #      pitch(상하 회전) → y 방향 이동
        dx = -self.fx * yaw_err * self.yaw_gain
        dy = -self.fy * pitch_err * self.pitch_gain

        # 4) 회전 + 줌 (마진 확보)
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, self.zoom)

        # 5) 평행이동 추가
        M[0, 2] += dx
        M[1, 2] += dy

        # 6) Affine 변환 적용
        stabilized = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 7) 중앙 크롭 + 원래 크기로 리사이즈
        crop_w = int(w * self.crop_ratio)
        crop_h = int(h * self.crop_ratio)
        sx = (w - crop_w) // 2
        sy = (h - crop_h) // 2

        cropped = stabilized[sy:sy + crop_h, sx:sx + crop_w]
        # 혹시라도 인덱스가 이상해지면 방어
        if cropped.size == 0:
            return stabilized

        final_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return final_frame

    # ==========================================
    # Image Callback → OpenCV Window
    # ==========================================
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        stabilized = self.stabilize_frame(frame)

        cv2.imshow(self.window_name, stabilized)
        # 1ms 대기 (키 입력 안 씀, 단순 갱신용)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = GimbalSoftNode()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
