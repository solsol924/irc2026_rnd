#!/usr/bin/env python3
import math
import rclpy
import cv2
import time

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class GimbalTrackingNode(Node):
    def __init__(self):
        super().__init__("gimbal_tracking")

        # ===== Parameters =====
        self.declare_parameter("alpha", 0.7)
        self.declare_parameter("cutoff_hz", 6)

        self.alpha = float(self.get_parameter("alpha").value)
        self.cutoff_hz = float(self.get_parameter("cutoff_hz").value)

        # --- raw absolute angle ---
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # --- absolute angle EMA ---
        self.roll_ema = 0.0
        self.pitch_ema = 0.0
        self.yaw_ema = 0.0

        # --- absolute angle LPF ---
        self.roll_lpf = 0.0
        self.pitch_lpf = 0.0
        self.yaw_lpf = 0.0

        # --- for jitter (delta based) ---
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0
        self.prev_time = time.time()

        # delta jitter filtering
        self.dpitch_ema = 0.0
        self.dyaw_ema = 0.0

        self.dpitch_lpf = 0.0
        self.dyaw_lpf = 0.0

        # camera params
        self.fx = 607.0
        self.fy = 606.0

        # history for RMS
        self.rms_window = 50
        self.jitter_raw_hist = []
        self.jitter_ema_hist = []
        self.jitter_lpf_hist = []

        self.delta_raw_hist = []
        self.delta_ema_hist = []
        self.delta_lpf_hist = []

        # publishers
        self.pub_jitter_raw = self.create_publisher(Float32, "/gimbal/jitter_raw", 10)
        self.pub_jitter_ema = self.create_publisher(Float32, "/gimbal/jitter_ema", 10)
        self.pub_jitter_lpf = self.create_publisher(Float32, "/gimbal/jitter_lpf", 10)

        self.pub_rms_raw = self.create_publisher(Float32, "/gimbal/rms_raw", 10)
        self.pub_rms_ema = self.create_publisher(Float32, "/gimbal/rms_ema", 10)
        self.pub_rms_lpf = self.create_publisher(Float32, "/gimbal/rms_lpf", 10)

        self.pub_delta_raw = self.create_publisher(Float32, "/gimbal/delta_raw", 10)
        self.pub_delta_ema = self.create_publisher(Float32, "/gimbal/delta_ema", 10)
        self.pub_delta_lpf = self.create_publisher(Float32, "/gimbal/delta_lpf", 10)    

        # ROS I/O
        self.bridge = CvBridge()
        self.create_subscription(Image, "/camera/color/image_raw", self.img_callback, 10)
        self.create_subscription(Vector3Stamped, "/camera/imu_tilt", self.imu_callback, 50)

        # windows
        cv2.namedWindow("Raw", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Stabilized_EMA", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Stabilized_LPF", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Raw", 640, 480)
        cv2.resizeWindow("Stabilized_EMA", 640, 480)
        cv2.resizeWindow("Stabilized_LPF", 640, 480)

        self.get_logger().info("GimbalTrackingNode started (correct jitter version)")

    # -------- jitter computation (pixel) --------
    def compute_jitter(self, dpitch, dyaw):
        dx = self.fx * dyaw
        dy = self.fy * dpitch
        return math.sqrt(dx*dx + dy*dy)

    def compute_rms(self, hist):
        if not hist:
            return 0.0
        s = sum(v*v for v in hist)
        return math.sqrt(s / len(hist))

    # -------- IMU callback --------
    def imu_callback(self, msg: Vector3Stamped):

        # absolute angles
        self.roll  = msg.vector.y
        self.pitch = msg.vector.x
        self.yaw   = msg.vector.z

        # ===== 1) STABILIZATION FILTERS (absolute angle) =====
        a = self.alpha

        # EMA
        self.roll_ema  = a*self.roll_ema  + (1-a)*self.roll
        self.pitch_ema = a*self.pitch_ema + (1-a)*self.pitch
        self.yaw_ema   = a*self.yaw_ema   + (1-a)*self.yaw

        # LPF
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            rc = 1.0 / (2 * math.pi * self.cutoff_hz)
            alpha = dt / (rc + dt)

            self.roll_lpf  += alpha * (self.roll  - self.roll_lpf)
            self.pitch_lpf += alpha * (self.pitch - self.pitch_lpf)
            self.yaw_lpf   += alpha * (self.yaw   - self.yaw_lpf)

        # ===== 2) JITTER FILTERS (delta angle) =====
        dpitch_raw = self.pitch - self.prev_pitch
        dyaw_raw   = self.yaw   - self.prev_yaw

        self.prev_pitch = self.pitch
        self.prev_yaw   = self.yaw

        # EMA(delta)
        self.dpitch_ema = a*self.dpitch_ema + (1-a)*dpitch_raw
        self.dyaw_ema   = a*self.dyaw_ema   + (1-a)*dyaw_raw

        # LPF(delta)
        if dt > 0:
            self.dpitch_lpf += alpha * (dpitch_raw - self.dpitch_lpf)
            self.dyaw_lpf   += alpha * (dyaw_raw   - self.dyaw_lpf)

        # ===== 3) jitter calculation =====
        jitter_raw = self.compute_jitter(dpitch_raw, dyaw_raw)
        jitter_ema = self.compute_jitter(self.dpitch_ema, self.dyaw_ema)
        jitter_lpf = self.compute_jitter(self.dpitch_lpf, self.dyaw_lpf)

        # ===== 4) RMS =====
        self.jitter_raw_hist.append(jitter_raw)
        self.jitter_ema_hist.append(jitter_ema)
        self.jitter_lpf_hist.append(jitter_lpf)

        delta_raw = self.compute_jitter(self.pitch, self.yaw)
        delta_ema = self.compute_jitter(self.pitch - self.pitch_ema, self.yaw - self.yaw_ema)
        delta_lpf = self.compute_jitter(self.pitch - self.pitch_lpf, self.yaw - self.yaw_lpf)

        self.delta_raw_hist.append(delta_raw)
        self.delta_ema_hist.append(delta_ema)
        self.delta_lpf_hist.append(delta_lpf)

        if len(self.jitter_raw_hist) > self.rms_window:
            self.jitter_raw_hist.pop(0)
            self.jitter_ema_hist.pop(0)
            self.jitter_lpf_hist.pop(0)

        if len(self.delta_raw_hist) > self.rms_window:
            self.delta_raw_hist.pop(0)
            self.delta_ema_hist.pop(0)
            self.delta_lpf_hist.pop(0)

        rms_raw = self.compute_rms(self.jitter_raw_hist)
        rms_ema = self.compute_rms(self.jitter_ema_hist)
        rms_lpf = self.compute_rms(self.jitter_lpf_hist)

        # publish
        self.pub_jitter_raw.publish(Float32(data=jitter_raw))
        self.pub_jitter_ema.publish(Float32(data=jitter_ema))
        self.pub_jitter_lpf.publish(Float32(data=jitter_lpf))

        self.pub_rms_raw.publish(Float32(data=rms_raw))
        self.pub_rms_ema.publish(Float32(data=rms_ema))
        self.pub_rms_lpf.publish(Float32(data=rms_lpf))

        self.pub_delta_raw.publish(Float32(data=delta_raw))
        self.pub_delta_ema.publish(Float32(data=delta_ema))
        self.pub_delta_lpf.publish(Float32(data=delta_lpf))

    # -------- stabilization transform (absolute angle) --------
    def stabilize_frame(self, frame, roll, pitch, yaw):

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        angle_deg = -roll * 180.0 / math.pi

        dx = -self.fx * yaw
        dy = -self.fy * pitch

        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        M[0,2] += dx
        M[1,2] += dy

        return cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0)
        )

    # -------- Image callback --------
    def img_callback(self, msg: Image):
        raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        ema = self.stabilize_frame(raw, self.roll_ema, self.pitch_ema, self.yaw_ema)
        lpf = self.stabilize_frame(raw, self.roll_lpf, self.pitch_lpf, self.yaw_lpf)

        cv2.imshow("Raw", raw)
        cv2.imshow("Stabilized_EMA", ema)
        cv2.imshow("Stabilized_LPF", lpf)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = GimbalTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
