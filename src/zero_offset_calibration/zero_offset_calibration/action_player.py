#!/usr/bin/env python3
"""动作数据播放器：加载 .data CSV 文件，通过上身调试接口以 100 Hz 播放。

用法::

    ros2 run zero_offset_calibration action_player --data <path.data>

.data 文件格式：第一行为表头（关节名含 ``_joint`` 后缀），后续每行为一帧关节角度（弧度），
以 100 Hz 播放（每帧 10 ms）。

播放时自动进入上身调试模式，播放完成后可选退出。
"""
from __future__ import annotations

import argparse
import csv
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState

try:
    from crb_ros_msg.msg import UpperJointData
except ImportError:
    raise SystemExit(
        "需要安装/编译 crb_ros_msg 包，使 Python 可 import crb_ros_msg.msg.UpperJointData。"
    )

logger = logging.getLogger(__name__)

# 上身关节关键词：含以下子串且以 _joint 结尾的列名会被发送到上身调试接口
_UPPER_BODY_KEYWORDS = (
    "shoulder", "elbow", "wrist",
    "head_yaw", "head_pitch", "waist_yaw",
    "thumb", "index", "middle", "ring", "pinky",
)


def _is_upper_body_joint(name: str) -> bool:
    if not name.endswith("_joint"):
        return False
    return any(kw in name for kw in _UPPER_BODY_KEYWORDS)


def _strip_joint_suffix(name: str) -> str:
    return name[:-len("_joint")] if name.endswith("_joint") else name


def load_action_data(path: Path) -> tuple[list[str], list[list[float]]]:
    """加载 .data CSV 文件，返回 (列名列表, 帧数据二维列表)。"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        frames: list[list[float]] = []
        for row in reader:
            vals = []
            for v in row:
                v = v.strip()
                if v == "":
                    continue
                vals.append(float(v))
            if vals:
                frames.append(vals)
    return header, frames


class ActionPlayer:
    """通过上身调试接口播放动作序列。"""

    def __init__(
        self,
        node: Node,
        *,
        service_debug: str = "/motion/upper_body_debug",
        topic_cmd: str = "/upper_body_debug/joint_cmd",
    ) -> None:
        self._node = node
        self._pub = node.create_publisher(UpperJointData, topic_cmd, 10)
        self._debug_cli = node.create_client(SetBool, service_debug)
        self._debug_enabled = False
        self._playing = False
        self._stop_event = threading.Event()

    def enable_debug(self, enable: bool, timeout_s: float = 5.0) -> bool:
        if not self._debug_cli.wait_for_service(timeout_sec=timeout_s):
            self._node.get_logger().error("上身调试服务不可用")
            return False
        req = SetBool.Request()
        req.data = enable
        future = self._debug_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_s)
        if future.result() is not None:
            self._debug_enabled = enable
            self._node.get_logger().info(
                "上身调试模式: %s" % ("开启" if enable else "关闭")
            )
            return True
        self._node.get_logger().error("上身调试模式请求失败")
        return False

    def play(
        self,
        header: list[str],
        frames: list[list[float]],
        hz: float = 100.0,
        vel_scale: float = 1.0,
        time_ref: float = 0.02,
    ) -> bool:
        """播放帧序列，返回是否完整播放（False=被中断）。"""
        col_indices: Dict[str, int] = {}
        for idx, name in enumerate(header):
            if _is_upper_body_joint(name):
                col_indices[name] = idx

        if not col_indices:
            self._node.get_logger().error("数据文件中无可用上身关节列")
            return False

        self._playing = True
        self._stop_event.clear()
        dt = 1.0 / hz
        total = len(frames)
        self._node.get_logger().info(
            "开始播放 %d 帧 @ %.0f Hz (%.1f 秒)" % (total, hz, total / hz)
        )

        for i, frame in enumerate(frames):
            if self._stop_event.is_set():
                self._node.get_logger().info("播放在第 %d/%d 帧被中断" % (i, total))
                self._playing = False
                return False

            ujd = UpperJointData()
            ujd.header = Header()
            ujd.header.stamp = self._node.get_clock().now().to_msg()
            ujd.time_ref = float(time_ref)
            ujd.vel_scale = float(vel_scale)

            js = JointState()
            for joint, idx in col_indices.items():
                if idx < len(frame):
                    ros_name = _strip_joint_suffix(joint)
                    js.name.append(ros_name)
                    js.position.append(frame[idx])
            ujd.joint = js
            self._pub.publish(ujd)

            time.sleep(dt)

        self._node.get_logger().info("播放完成 (%d 帧)" % total)
        self._playing = False
        return True

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def is_playing(self) -> bool:
        return self._playing


def main() -> int:
    rclpy.init()
    node = Node("casbot_action_player")

    ap = argparse.ArgumentParser(description="CASBOT02 动作数据播放器")
    ap.add_argument("--data", type=Path, required=True, help=".data 文件路径")
    ap.add_argument("--hz", type=float, default=100.0, help="播放频率 (默认 100)")
    ap.add_argument(
        "--no-exit-debug", action="store_true",
        help="播放完成后不退出上身调试模式",
    )
    args = ap.parse_args()

    if not args.data.is_file():
        node.get_logger().error("文件不存在: %s" % args.data)
        return 1

    header, frames = load_action_data(args.data)
    node.get_logger().info("已加载 %s: %d 帧, %d 列" % (args.data.name, len(frames), len(header)))

    player = ActionPlayer(node)

    if not player.enable_debug(True):
        node.destroy_node()
        rclpy.shutdown()
        return 1

    try:
        player.play(header, frames, hz=args.hz)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
        player.stop()
    finally:
        if not args.no_exit_debug:
            player.enable_debug(False)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
