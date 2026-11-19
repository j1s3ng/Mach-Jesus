"""Optional ROS 2 bridge. Run in an environment with rclpy + vision_msgs.

Clouds must already be in the configured ego ground frame. This node deliberately
does not silently relabel a sensor-local cloud without a TF transformation.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
    from race_lidar.baseline import detect
    from race_lidar.remote import remote_detect

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="ego_ground")
    parser.add_argument("--detector-url", default="")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)

    class OpponentNode(Node):
        def __init__(self):
            super().__init__("race_lidar_opponents")
            self.publisher = self.create_publisher(Detection3DArray, "/perception/opponents", 10)
            self.subscription = self.create_subscription(PointCloud2, "/lidar/points", self.on_cloud, qos_profile_sensor_data)

        def on_cloud(self, message):
            if message.header.frame_id != args.frame:
                self.get_logger().error(f"Expected cloud transformed to {args.frame}, received {message.header.frame_id}")
                return
            names = [field.name for field in message.fields]
            if not all(name in names for name in ("x", "y", "z")):
                self.get_logger().error("Cloud needs x,y,z fields")
                return
            fields = ["x", "y", "z"] + (["intensity"] if "intensity" in names else [])
            rows = point_cloud2.read_points(message, field_names=fields, skip_nans=True)
            if isinstance(rows, np.ndarray) and rows.dtype.names:
                points = np.column_stack([rows[name].reshape(-1) for name in fields]).astype(np.float32)
            else:
                points = np.asarray(list(rows), dtype=np.float32).reshape(-1,len(fields))
            if len(fields) == 3:
                points = np.column_stack([points,np.ones(len(points),dtype=np.float32)])
            boxes = remote_detect(points,args.detector_url) if args.detector_url else detect(points)
            output = Detection3DArray()
            output.header = message.header
            for box in boxes:
                detection = Detection3D()
                detection.header = message.header
                pose = detection.bbox.center
                pose.position.x, pose.position.y, pose.position.z = map(float,box["center"])
                pose.orientation.z = math.sin(box["yaw"]/2)
                pose.orientation.w = math.cos(box["yaw"]/2)
                detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = map(float,box["size"])
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = "car" if args.detector_url else "vehicle_sized_candidate"
                hypothesis.hypothesis.score = float(box["score"])
                detection.results.append(hypothesis)
                output.detections.append(detection)
            self.publisher.publish(output)

    node = OpponentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
