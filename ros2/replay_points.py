"""Publish a portable recording to /lidar/points with original timestamps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from sensor_msgs_py import point_cloud2
    from race_lidar.io import read_frame, read_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--frame", default="ego_ground")
    parser.add_argument("--hz", type=float, default=10)
    args, ros_args = parser.parse_known_args()
    if args.hz <= 0: parser.error("--hz must be positive")
    entries = read_manifest(args.recording)
    rclpy.init(args=ros_args)
    node = Node("race_lidar_replay")
    publisher = node.create_publisher(PointCloud2,"/lidar/points",qos_profile_sensor_data)
    fields = [PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1)
              for i,name in enumerate(["x","y","z","intensity"])]
    pending = iter(entries)
    def publish():
        entry = next(pending,None)
        if entry is None:
            rclpy.shutdown()
            return
        frame = read_frame(entry,args.recording)
        header = Header()
        header.frame_id = args.frame
        header.stamp.sec, header.stamp.nanosec = divmod(entry["timestamp_ns"],1_000_000_000)
        publisher.publish(point_cloud2.create_cloud(header,fields,frame["points"]))
    timer = node.create_timer(1/args.hz,publish)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):
        pass
    finally:
        timer.cancel()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
