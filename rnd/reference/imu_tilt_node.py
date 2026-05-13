#!/usr/bin/env python3
import math
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from rclpy.qos import qos_profile_sensor_data



class ImuTiltNode(Node):
    def __init__(self):
        super().__init__('imu_tilt_node')

        self.declare_parameter('gyro_topic', '/camera/gyro/sample')
        self.declare_parameter('accel_topic', '/camera/accel/sample')
        self.declare_parameter('calib_duration', 10.0)   # 바이어스 측정
        self.declare_parameter('alpha', 0.97)           # 자이로-가속도계 비율

        self.gyro_topic = self.get_parameter('gyro_topic').get_parameter_value().string_value
        self.accel_topic = self.get_parameter('accel_topic').get_parameter_value().string_value
        self.calib_duration = self.get_parameter('calib_duration').get_parameter_value().double_value
        self.alpha = self.get_parameter('alpha').get_parameter_value().double_value

        self.last_gyro = None
        self.last_accel = None
        self.last_time = None

        self.roll_offset = 0.0
        self.pitch_offset = 0.0

        self.calibrating = True
        self.calib_start_time = None
        self.bias_x = 0.0
        self.bias_y = 0.0
        self.bias_z = 0.0
        self.bias_samples = 0

        self.roll = 0.0
        self.pitch = 0.0

        # 퍼블리시
        self.sub_gyro = self.create_subscription(
            Imu, self.gyro_topic, self.gyro_callback, qos_profile_sensor_data)

        self.sub_accel = self.create_subscription(
            Imu, self.accel_topic, self.accel_callback, qos_profile_sensor_data)


        self.pub_tilt = self.create_publisher(
            Vector3Stamped, '/camera/imu_tilt', 10)

        # 200Hz 기준 타이머 (IMU 주파수에 맞춰)
        self.timer = self.create_timer(0.005, self.update_filter)

        self.get_logger().info('IMU Tilt Node started.')

    def gyro_callback(self, msg: Imu):
        self.last_gyro = msg

        if self.calibrating and self.calib_start_time is None:
            self.calib_start_time = self.get_clock().now()

    def accel_callback(self, msg: Imu):
        self.last_accel = msg

    def update_filter(self):
        if self.last_gyro is None or self.last_accel is None:
            return

        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        gx = self.last_gyro.angular_velocity.x
        gy = self.last_gyro.angular_velocity.y
        gz = self.last_gyro.angular_velocity.z

        ax = self.last_accel.linear_acceleration.x
        ay = self.last_accel.linear_acceleration.y
        az = self.last_accel.linear_acceleration.z

        # ---------- 1) 초기 바이어스 측정 ----------
        if self.calibrating:
            elapsed = (now - self.calib_start_time).nanoseconds * 1e-9
            if elapsed < self.calib_duration:
                # 계속 평균 내기
                self.bias_x += gx
                self.bias_y += gy
                self.bias_z += gz
                self.bias_samples += 1
                return
            else:
                # 평균값으로 bias 확정
                if self.bias_samples > 0:
                    self.bias_x /= self.bias_samples
                    self.bias_y /= self.bias_samples
                    self.bias_z /= self.bias_samples
                self.calibrating = False
                self.get_logger().info(
                    f'Gyro bias calibrated: '
                    f'bx={self.bias_x:.6f}, by={self.bias_y:.6f}, bz={self.bias_z:.6f}')

                # 이 순간의 자세를 "기준(0도)"로 사용하기 위해 offset 저장
                roll0, pitch0 = self.compute_tilt_from_accel(ax, ay, az)
                self.roll_offset = roll0
                self.pitch_offset = pitch0

                # 기준 자세에서는 roll = 0, pitch = 0 으로 시작
                self.roll = 0.0
                self.pitch = 0.0

                self.get_logger().info(
                    f'Initial tilt offset: roll0={roll0:.3f}, pitch0={pitch0:.3f}')
                return


        # ---------- 2) 바이어스 보정된 gyro 사용 ----------
        gx -= self.bias_x
        gy -= self.bias_y
        gz -= self.bias_z

        # 여기서는 예시로:
        #  - gx: roll rate (rad/s)
        #  - gy: pitch rate (rad/s)
        # 축 정의는 설치 방향에 따라 바뀔 수 있음
        roll_gyro  = self.roll  + gx * dt
        pitch_gyro = self.pitch + gy * dt

        # ---------- 3) accel로부터 roll/pitch 계산 ----------
        roll_acc, pitch_acc = self.compute_tilt_from_accel(ax, ay, az)

        # 초기 자세(roll_offset, pitch_offset)를 0도로 만들기
        roll_acc  -= self.roll_offset
        pitch_acc -= self.pitch_offset


        # ---------- 4) complementary filter ----------
        a = self.alpha
        self.roll  = a * roll_gyro  + (1.0 - a) * roll_acc
        self.pitch = a * pitch_gyro + (1.0 - a) * pitch_acc

        # ---------- 5) publish ----------
        msg_out = Vector3Stamped()
        msg_out.header.stamp = now.to_msg()
        msg_out.header.frame_id = 'camera_link'
        msg_out.vector.x = self.roll
        msg_out.vector.y = self.pitch
        msg_out.vector.z = 0.0  # yaw는 여기선 안 씀
        self.pub_tilt.publish(msg_out)

    @staticmethod
    def compute_tilt_from_accel(ax, ay, az):
        """
        가속도에서 roll, pitch 추정 (rad)
        많이 쓰는 근사식:
          roll  = atan2( ay, az )
          pitch = atan2( -ax, sqrt(ay^2 + az^2) )
        """
        # 작은 값으로 0 나눔 방지
        eps = 1e-6
        roll = math.atan2(ay, az if abs(az) > eps else eps)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        return roll, pitch


def main():
    rclpy.init()
    node = ImuTiltNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
