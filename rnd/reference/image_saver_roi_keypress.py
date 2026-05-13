import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob

class ImageSaverROIKey(Node):
    def __init__(self):
        super().__init__('image_saver_roi_key')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        self.frame = None
        self.image_count = 0
        self.save_dir = os.path.abspath('dataset/images/train')

        os.makedirs(self.save_dir, exist_ok=True)

        existing_imgs = sorted(glob.glob(os.path.join(self.save_dir, 'img_*.jpg')))
        if existing_imgs:
            last_img = existing_imgs[-1]
            last_num = int(os.path.basename(last_img).split('_')[1].split('.')[0])
            self.image_count = last_num + 1
        else:
            self.image_count = 0

        # ROI: 중앙 320x480
        self.roi_x = 160
        self.roi_y = 0
        self.roi_w = 320
        self.roi_h = 480

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def main():
    rclpy.init()
    node = ImageSaverROIKey()

    print("Press SPACE to save image, ESC to quit.")
    cv2.namedWindow("ROI Viewer", cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.frame is not None:
                roi = node.frame[
                    node.roi_y:node.roi_y + node.roi_h,
                    node.roi_x:node.roi_x + node.roi_w
                ]

                cv2.imshow("ROI Viewer", roi)
                key = cv2.waitKey(10)

                if key == 27:  # ESC key
                    break
                elif key == 32:  # SPACE key
                    filename = os.path.join(node.save_dir, f"img_{node.image_count:04}.jpg")
                    cv2.imwrite(filename, roi)
                    print(f"Saved: {filename}")
                    node.image_count += 1

    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
