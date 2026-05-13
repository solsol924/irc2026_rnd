import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

class BallListenerNode(Node):
    def __init__(self):
        super().__init__('ball_listener')
        self.sub = self.create_subscription(
            PointStamped,                   # 메시지 타입
            '/basketball/position',         # 토픽 이름
            self.ball_callback,             # 콜백 함수
            10                              # 큐 깊이
        )

    def ball_callback(self, msg: PointStamped): 
        # 좌표 데이터 추출
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        # 로그로 좌표 출력 (터미널이나 rqt_console에서 확인 가능)
        self.get_logger().info(f'Ball at X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m')

def main():
    rclpy.init()
    node = BallListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
