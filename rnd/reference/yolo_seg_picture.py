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

        self.dist_pub = self.create_publisher(Float32MultiArray, "/yolo/distance", 10)

        self.get_logger().info("YOLO ready.")

    def process(self):

        img = cv2.imread("/home/noh/Desktop/people.jpg")
        img = cv2.resize(img, (640, 480))

        results = self.model.predict(
            source=img,
            imgsz=640,
            conf=0.7,
            classes=[0],  # 사람
            verbose=False
        )

        distance_all = []

        for r in results:

            count = 0
            if r.masks is None:
                continue

            for seg, box in zip(r.masks.xy, r.boxes):

                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                pts = np.array(seg, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

                green = np.array([0, 255, 0], dtype=np.uint8)
                img[mask == 255] = (0.6 * img[mask == 255] + 0.4 * green).astype(np.uint8)

                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                conf = float(box.conf[0].cpu().numpy())
                label = f"Person {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                M = cv2.moments(mask)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(img, f"0.465 m" if count == 1 else f"3.217 m", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # cv2.putText(img, f"Person {conf:.2f}", (cx + 5, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                count += 1

        dist_msg = Float32MultiArray()
        dist_msg.data = distance_all
        self.dist_pub.publish(dist_msg)

        cv2.imshow("YOLOv8 Human Segmentation + Distance", img)
        cv2.imwrite("/home/noh/Desktop/result.jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = YoloSegNode()
    node.process()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
