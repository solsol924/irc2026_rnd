#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from rclpy.qos import qos_profile_sensor_data

import numpy as np


def normalize(v): # 형식: np.ndarray
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    return v / n


def rotation_matrix_from_to(a, b): # a를 b로 회전하는 함수 (단위행렬만)
    a = normalize(a)
    b = normalize(b)

    v = np.cross(a, b) # 외적
    c = np.dot(a, b)   # 내적

    if np.linalg.norm(v) < 1e-6: # 방향 맞추기
        if c > 0:
            return np.eye(3)
        else:
            return np.array([
                [1.0, 0.0,  0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])

    # skew-symmetric matrix [v]_x
    vx = np.array([
        [0.0,    -v[2],  v[1]],
        [v[2],    0.0,  -v[0]],
        [-v[1],   v[0],  0.0]
    ])

    # Rodrigues의 공식 변형: R = I + vx + vx^2 / (1 + c)
    R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return R


class ImuSampleNode(Node):
    def __init__(self):
        super().__init__('imu_sample_node')

        # ===== 파라미터 =====
        self.declare_parameter('calib_duration', 5.0)

        self.declare_parameter('use_ema', True) # 이동평균
        self.declare_parameter('use_lpf', False) # 로우패스

        self.declare_parameter('alpha_acc', 0.95) # 작을수록 민감
        self.declare_parameter('alpha_gyro', 0.90) # 작을수록 민감
        self.declare_parameter('lpf_tau', 0.05)   # 작을수록 민감 < 단위는 초
        self.declare_parameter('ori_alpha', 0.98) # 상보필터에서 gyro 비율

        self.calib_duration = float(self.get_parameter('calib_duration').value)

        self.use_ema = bool(self.get_parameter('use_ema').value)
        self.use_lpf = bool(self.get_parameter('use_lpf').value)
        self.alpha_acc = float(self.get_parameter('alpha_acc').value)
        self.alpha_gyro = float(self.get_parameter('alpha_gyro').value)
        self.lpf_tau = float(self.get_parameter('lpf_tau').value)
        self.ori_alpha = float(self.get_parameter('ori_alpha').value)

        # ===== 상태 변수 =====
        self.last_gyro: Imu | None = None
        self.last_accel: Imu | None = None

        self.calibrating = True
        self.calib_start_time = None
        self.bias_samples = 0

        # 자이로 bias 합
        self.gyro_bias_sum = np.zeros(3, dtype=float)

        # 가속도 합 (중력 방향 추정용)
        self.accel_sum = np.zeros(3, dtype=float)

        # 최종 바이어스 & 회전행렬
        self.gyro_bias = np.zeros(3, dtype=float)
        self.R0 = np.eye(3, dtype=float)

        # 시간 추정을 위한 이전 시각
        self.last_time = None

        # ===== 필터 내부 상태 =====
        # EMA
        self.accel_ema = np.zeros(3, dtype=float)
        self.gyro_ema = np.zeros(3, dtype=float)

        # 1차 LPF
        self.accel_lpf = np.zeros(3, dtype=float)
        self.gyro_lpf = np.zeros(3, dtype=float)

        # 보완필터로 추정한 roll, pitch (라디안)
        self.roll = 0.0
        self.pitch = 0.0

        # ===== 구독자 =====
        self.sub_gyro = self.create_subscription(Imu, '/camera/gyro/sample', self.gyro_callback, qos_profile_sensor_data)
        self.sub_accel = self.create_subscription(Imu, '/camera/accel/sample', self.accel_callback, qos_profile_sensor_data)

        # ===== 퍼블리셔 =====
        # R0까지만 적용된 값 (레벨링만 된 값)
        self.pub_rot_gyro = self.create_publisher(
            Vector3Stamped, 'imu/gyro_leveled', 10)
        self.pub_rot_accel = self.create_publisher(
            Vector3Stamped, 'imu/accel_leveled', 10)

        # 필터까지 적용된 값
        self.pub_filt_gyro = self.create_publisher(
            Vector3Stamped, 'imu/gyro_filtered', 10)
        self.pub_filt_accel = self.create_publisher(
            Vector3Stamped, 'imu/accel_filtered', 10)

        # 보완필터로 추정한 기울기 (roll, pitch)
        self.pub_tilt = self.create_publisher(
            Vector3Stamped, 'imu/tilt', 10)

        # ===== 타이머 루프 (약 200Hz) =====
        self.timer = self.create_timer(0.005, self.process)

        self.get_logger().info(
            f"IMU sample node started. calib_duration={self.calib_duration:.2f}s, "
            f"use_ema={self.use_ema}, use_lpf={self.use_lpf}"
        )

    # ---------- 콜백들 ----------

    def gyro_callback(self, msg: Imu):
        self.last_gyro = msg
        if self.calibrating and self.calib_start_time is None:
            self.calib_start_time = self.get_clock().now()

    def accel_callback(self, msg: Imu):
        self.last_accel = msg

    # ---------- 메인 처리 루프 ----------

    def process(self):
        # 아직 센서 데이터가 준비 안 되었으면 대기
        if self.last_gyro is None or self.last_accel is None:
            return

        now = self.get_clock().now()

        # 초기 센서 값 추출
        wx = self.last_gyro.angular_velocity.x
        wy = self.last_gyro.angular_velocity.y
        wz = self.last_gyro.angular_velocity.z

        ax = self.last_accel.linear_acceleration.x
        ay = self.last_accel.linear_acceleration.y
        az = self.last_accel.linear_acceleration.z

        gyro_raw = np.array([wx, wy, wz], dtype=float)
        accel_raw = np.array([ax, ay, az], dtype=float)

        # ===== (1) 캘리브레이션 단계 =====
        if self.calibrating:
            if self.calib_start_time is None:
                # 아직 자이로 콜백에서 시작 시간이 설정되지 않은 경우
                return

            elapsed = (now - self.calib_start_time).nanoseconds * 1e-9
            if elapsed < self.calib_duration:
                # 자이로 바이어스 및 중력 방향 추정을 위한 합 계수
                self.gyro_bias_sum += gyro_raw
                self.accel_sum += accel_raw
                self.bias_samples += 1
                return
            else:
                # 평균 계산
                if self.bias_samples > 0:
                    self.gyro_bias = self.gyro_bias_sum / self.bias_samples
                    g0 = self.accel_sum / self.bias_samples
                else:
                    self.get_logger().warn("No samples collected during calibration.")
                    self.gyro_bias = np.zeros(3, dtype=float)
                    g0 = np.array([0.0, -1.0, 0.0], dtype=float)

                g0 = normalize(g0)
                new_gravity = np.array([0.0, -1.0, 0.0], dtype=float)

                # 회전 행렬 R0 계산 (g0 -> new_gravity)
                self.R0 = rotation_matrix_from_to(g0, new_gravity)

                self.calibrating = False
                self.last_time = now  # 이후 dt 계산 시작 지점

                self.get_logger().info(f"Gyro bias = {self.gyro_bias}")
                self.get_logger().info(f"Initial gravity direction g0 = {g0}")
                self.get_logger().info(f"Rotation matrix R0:\n{self.R0}")
                return

        # ===== (2) 5초 5후 =====

        # dt 계산
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # dt가 너무 비정상적으로 크거나 작으면 무시
        if dt <= 0.0 or dt > 0.1:
            # 타이밍 문제로 인한 이상값 방지
            return

        # 자이로 바이어스 제거 후, R0 적용 (레벨링)
        gyro_corr = self.R0 @ (gyro_raw - self.gyro_bias)
        accel_corr = self.R0 @ accel_raw  # 가속도는 바이어스 제거 없이 그대로 회전만

        # ===== (2-1) leveled 값 퍼블리시 =====
        msg_g_lev = Vector3Stamped()
        msg_g_lev.header.stamp = now.to_msg()
        msg_g_lev.vector.x = float(gyro_corr[0])
        msg_g_lev.vector.y = float(gyro_corr[1])
        msg_g_lev.vector.z = float(gyro_corr[2])
        self.pub_rot_gyro.publish(msg_g_lev)

        msg_a_lev = Vector3Stamped()
        msg_a_lev.header.stamp = now.to_msg()
        msg_a_lev.vector.x = float(accel_corr[0])
        msg_a_lev.vector.y = float(accel_corr[1])
        msg_a_lev.vector.z = float(accel_corr[2])
        self.pub_rot_accel.publish(msg_a_lev)

        # ===== (3) EMA 필터 =====
        if self.use_ema:
            self.accel_ema = self.alpha_acc * self.accel_ema + (1.0 - self.alpha_acc) * accel_corr
            self.gyro_ema  = self.alpha_gyro * self.gyro_ema  + (1.0 - self.alpha_gyro)  * gyro_corr
        else:
            self.accel_ema = accel_corr
            self.gyro_ema  = gyro_corr

        # ===== (4) 1차 Low-pass 필터 (옵션) =====
        if self.use_lpf:
            alpha_lpf = dt / (self.lpf_tau + dt)
            self.accel_lpf = self.accel_lpf + alpha_lpf * (self.accel_ema - self.accel_lpf)
            self.gyro_lpf  = self.gyro_lpf  + alpha_lpf * (self.gyro_ema  - self.gyro_lpf)

            accel_out = self.accel_lpf
            gyro_out  = self.gyro_lpf
        else:
            accel_out = self.accel_ema
            gyro_out  = self.gyro_ema

        # ===== (4-1) 필터 결과 퍼블리시 =====
        msg_g_f = Vector3Stamped()
        msg_g_f.header.stamp = now.to_msg()
        msg_g_f.vector.x = float(gyro_out[0])
        msg_g_f.vector.y = float(gyro_out[1])
        msg_g_f.vector.z = float(gyro_out[2])
        self.pub_filt_gyro.publish(msg_g_f)

        msg_a_f = Vector3Stamped()
        msg_a_f.header.stamp = now.to_msg()
        msg_a_f.vector.x = float(accel_out[0])
        msg_a_f.vector.y = float(accel_out[1])
        msg_a_f.vector.z = float(accel_out[2])
        self.pub_filt_accel.publish(msg_a_f)

        # ===== (5) 보완필터로 roll, pitch 계산 =====
        # accel 기반 roll/pitch (라디안)
        ay_f, az_f = accel_out[1], accel_out[2]
        ax_f = accel_out[0]

        # 안전하게 0 나눔 방지
        if abs(az_f) < 1e-6 and abs(ay_f) < 1e-6:
            roll_acc = self.roll
            pitch_acc = self.pitch
        else:
            roll_acc = math.atan2(ay_f, az_f)
            pitch_acc = math.atan2(-ax_f, math.sqrt(ay_f * ay_f + az_f * az_f))

        # 자이로 적분으로 예측
        roll_gyro  = self.roll  + gyro_out[0] * dt
        pitch_gyro = self.pitch + gyro_out[1] * dt   # 축 정의에 따라 변경 가능

        # 보완필터 (gyro 비중: ori_alpha, accel 비중: 1 - ori_alpha)
        a = self.ori_alpha
        self.roll  = a * roll_gyro  + (1.0 - a) * roll_acc
        self.pitch = a * pitch_gyro + (1.0 - a) * pitch_acc

        # ===== (5-1) 기울기 퍼블리시 =====
        msg_tilt = Vector3Stamped()
        msg_tilt.header.stamp = now.to_msg()
        msg_tilt.vector.x = self.roll   # roll (rad)
        msg_tilt.vector.y = self.pitch  # pitch (rad)
        msg_tilt.vector.z = 0.0         # yaw는 여기서는 계산하지 않음
        self.pub_tilt.publish(msg_tilt)


def main():
    rclpy.init()
    node = ImuSampleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
