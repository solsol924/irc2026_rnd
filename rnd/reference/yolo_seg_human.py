#!/usr/bin/env python3
import rclpy
import torch
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer

class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')
        self.get_logger().info("Loading YOLOv8n-seg model...")

        self.model = YOLO("yolov8n-seg.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.get_logger().info(f"Model device: {device}")

        self.bridge = CvBridge()

        self.color_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw")

        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1).registerCallback(self.image_callback)

        self.dist_pub = self.create_publisher(Float32MultiArray, "/yolo/distance", 10)
        self.image_pub = self.create_publisher(Image, "/yolo/segmented_image", 10)

        self.get_logger().info("YOLO ready.")

    def image_callback(self, color_msg: Image, depth_msg: Image):

        color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)

        results = self.model.predict(
            source=color,
            imgsz=640,
            conf=0.7,
            classes=[0],  # 사람
            verbose=False
        )

        color_seg = color.copy()

        distance_all = []

        for r in results:
            if r.masks is None:
                continue

            for seg, box in zip(r.masks.xy, r.boxes):

                mask = np.zeros(color.shape[:2], dtype=np.uint8)
                pts = np.array(seg, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

                green = np.array([0, 255, 0], dtype=np.uint8)
                color_seg[mask == 255] = (0.6 * color_seg[mask == 255] + 0.4 * green).astype(np.uint8)

                # coords = box.xyxy[0].cpu().numpy().astype(int)
                # x1, y1, x2, y2 = coords
                conf = float(box.conf[0].cpu().numpy())
                # # label = f"Person {conf:.2f}"
                # cv2.rectangle(color_seg, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # cv2.putText(color_seg, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                M = cv2.moments(mask)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                h, w = depth.shape[:2]

                x_min = max(cx - 2, 0)
                x_max = min(cx + 3, w)
                y_min = max(cy - 2, 0)
                y_max = min(cy + 3, h)

                depth_window = depth[y_min:y_max, x_min:x_max]
                depth_valid = depth_window[depth_window > 0]

                distance = None
                if depth_valid.size > 0:
                    distance = float(np.median(depth_valid)) * 0.001   

                cv2.circle(color_seg, (cx, cy), 4, (0, 0, 255), -1)

                if distance is not None:
                    distance_all.append(distance)
                    cv2.putText(color_seg, f"{distance:.3f} m", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(color_seg, f"Person {conf:.2f}", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    distance_all.append(-1.0)

        dist_msg = Float32MultiArray()
        dist_msg.data = distance_all
        self.dist_pub.publish(dist_msg)

        cv2.imshow("YOLOv8 Human Segmentation + Distance", color_seg)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = YoloSegNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
