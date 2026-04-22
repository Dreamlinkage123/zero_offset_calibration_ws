#!/usr/bin/env python3
"""ROS 2 上身调试接口真机适配（依据《CASBOT02 二次开发文档-上半身》）.

- 3.2 上身关节控制
  - Service: /motion/upper_body_debug  (std_srvs/srv/SetBool)  进入/退出上身调试
  - Topic:   /upper_body_debug/joint_cmd  (crb_ros_msg/msg/UpperJointData)  关节空间位置控制
- 2.5.3 关节状态：官方约定 `sensor_msgs/msg/JointState`，topic 为 /joint_states（含腿、腰、
  头、双臂、灵巧手等关节）。本实现默认订阅该名；若 remapping/重名，用参数覆盖即可。
- 2.5.2 IMU：/imu（sensor_msgs/msg/Imu），本零偏校准则可不使用，仅作接口索引。

从 /joint_states 取 position/velocity/effort；effort 的绝对值参与硬限位“电流/力矩”判据。

`search_mode=torque_damping` 时：在仅有位置流的前提下，用 τ_cmd=sign·T−b·q̇ 换算成目标角速度再积分到 `q_cmd`（`Ros2UpperBodyConfig.torque2vel`），近似恒力矩+阻尼；若日后有力矩/电流内环可改为直接发矩。

零偏：文档仍无专门零偏写入服务。`--persist` 时用 `write_zero_offsets_yaml()` 将测算结果
写入 `--offsets-out` 指定的 **YAML 文件**（不依赖 PyYAML 包）。
`apply_zero_offset` 在内存中仅打日志，完整结果在 `calibrate` 结束一次性落盘。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, cast

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState

try:
    from crb_ros_msg.msg import UpperJointData
except ImportError as exc:  # pragma: no cover - 仅在实机/工作空间有包时存在
    raise ImportError(
        "需要安装/编译 crb_ros_msg 包，使 Python 可 import crb_ros_msg.msg.UpperJointData。"
    ) from exc

from hard_stop_calibration import (
    CalibrationHardware,
    HardStopCalibrator,
    HardStopCalibratorConfig,
    JointLimit,
    JointSample,
    arm_joint_names,
    build_default_arm_calibration_plan,
    parse_joint_limits,
    plan_summary,
    write_zero_offsets_yaml,
)


# 3.2 节列出的上身+灵巧手关节名顺序（与 UpperJointData 中 joint 的组装一致）。
UPPER_JOINTS_DOC_ORDER: List[str] = [
    "head_yaw_joint",
    "head_pitch_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
]


@dataclass
class Ros2UpperBodyConfig:
    urdf_path: Path
    service_debug: str = "/motion/upper_body_debug"
    topic_joint_cmd: str = "/upper_body_debug/joint_cmd"
    topic_joint_states: str = "/joint_states"
    time_ref_move_s: float = 2.0
    time_ref_search_step_s: float = 0.2
    search_period_s: float = 0.02
    move_tolerance_rad: float = 0.02
    move_timeout_s: float = 40.0
    offsets_file: Path = field(default_factory=lambda: Path("zero_offsets.yaml"))
    # 仅力矩+阻尼找限位：UpperJointData 为位置流，用 τ ≈ b·(q̇_ref−q̇) 的等价目标速度 (rad/s) / (N·m)
    torque2vel: float = 0.012
    torque_search_vel_max: float = 0.12


class UpperBodyDebugHardware:
    """实现 CalibrationHardware，对接 CASBOT02 上身调试 topic。"""

    def __init__(self, node: Node, cfg: Ros2UpperBodyConfig, arm_joint_names: tuple[str, ...]) -> None:
        self._node = node
        self._cfg = cfg
        self._arm_joints = arm_joint_names
        self._limits: Dict[str, JointLimit] = {
            j.name: j for j in parse_joint_limits(cfg.urdf_path).values() if j.name in UPPER_JOINTS_DOC_ORDER
        }
        for name in arm_joint_names:
            if name not in self._limits:
                raise KeyError(f"URDF 中未找到关节 {name}，请检查 {cfg.urdf_path}")

        self._lock = threading.Lock()
        self._state_pos: Dict[str, float] = {}
        self._state_vel: Dict[str, float] = {}
        self._state_eff: Dict[str, float] = {}
        self._state_stamp: float = 0.0
        self._cmd_positions: Dict[str, float] = {}

        self._search_thread: Optional[threading.Thread] = None
        self._search_stop = threading.Event()
        self._search_joint: Optional[str] = None
        self._search_vel: float = 0.0
        # "velocity" | "torque"
        self._search_mode: str = "velocity"
        self._td_sign: float = 1.0
        self._td_tau: float = 0.0
        self._td_damp: float = 0.0

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self._node.create_publisher(UpperJointData, cfg.topic_joint_cmd, 10)
        self._sub = self._node.create_subscription(
            JointState, cfg.topic_joint_states, self._on_joint_state, qos
        )
        self._debug_client = self._node.create_client(SetBool, cfg.service_debug)
        self._debug_enabled = False

    def enable_upper_body_debug(self, enable: bool, timeout_s: float = 10.0) -> bool:
        if not self._debug_client.wait_for_service(timeout_sec=timeout_s):
            self._node.get_logger().error("上身调试服务不可用: %s" % self._cfg.service_debug)
            return False
        req = SetBool.Request()
        req.data = enable
        future = self._debug_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_s)
        if future.done() and future.result() is not None and future.result().success:
            self._debug_enabled = bool(enable)
            self._node.get_logger().info("上身调试模式: %s" % ("开启" if enable else "关闭"))
            return True
        self._node.get_logger().error("调用上身调试服务失败或 success=false")
        return False

    def _on_joint_state(self, msg: JointState) -> None:
        now = time.monotonic()
        with self._lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self._state_pos[name] = float(msg.position[i])
                if i < len(msg.velocity):
                    self._state_vel[name] = float(msg.velocity[i])
                else:
                    self._state_vel[name] = 0.0
                if i < len(msg.effort):
                    self._state_eff[name] = float(msg.effort[i])
                else:
                    self._state_eff[name] = 0.0
            self._state_stamp = now
        # 首次用反馈初始化指令缓存，使未在 pose 中的关节保持当前位置
        if not self._cmd_positions:
            for name, pos in self._state_pos.items():
                self._cmd_positions[name] = pos

    def _ensure_cmd_seed(self) -> None:
        for name in UPPER_JOINTS_DOC_ORDER:
            if name in self._state_pos and name not in self._cmd_positions:
                self._cmd_positions[name] = self._state_pos[name]

    def _joint_state_msg_from_cmd(self) -> JointState:
        self._ensure_cmd_seed()
        js = JointState()
        for name in UPPER_JOINTS_DOC_ORDER:
            if name in self._cmd_positions:
                pos = self._cmd_positions[name]
            elif name in self._state_pos:
                pos = self._state_pos[name]
            else:
                continue
            js.name.append(name)
            js.position.append(pos)
        return js

    def _publish_cmd(self, vel_scale: float, time_ref: float) -> None:
        ujd = UpperJointData()
        ujd.header = Header()
        ujd.header.stamp = self._node.get_clock().now().to_msg()
        ujd.time_ref = float(time_ref)
        ujd.vel_scale = float(max(0.0, min(1.0, vel_scale)))
        ujd.joint = self._joint_state_msg_from_cmd()
        self._pub.publish(ujd)

    def _spin_until_joint_state(self, timeout_s: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            with self._lock:
                if self._arm_joints and all(j in self._state_pos for j in self._arm_joints):
                    return True
        return False

    def move_to_pose(self, pose: Dict[str, float], speed_scale: float) -> None:
        if not self._debug_enabled:
            raise RuntimeError("请先调用 enable_upper_body_debug(true)")
        if not self._spin_until_joint_state(5.0):
            self._node.get_logger().warn("未在超时内收齐 /joint_states，将仍尝试发送指令")
        for k, v in pose.items():
            self._cmd_positions[k] = float(v)
        self._publish_cmd(vel_scale=speed_scale, time_ref=self._cfg.time_ref_move_s)
        # 等待进入容差
        t0 = time.monotonic()
        while time.monotonic() - t0 < self._cfg.move_timeout_s:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            with self._lock:
                ok = all(
                    abs(self._state_pos.get(j, 0.0) - float(pose[j])) < self._cfg.move_tolerance_rad
                    for j in pose
                )
            if ok:
                return
        raise TimeoutError("move_to_pose 超时: %s" % (list(pose.keys()),))

    def read_sample(self, joint_name: str) -> JointSample:
        rclpy.spin_once(self._node, timeout_sec=0.02)
        with self._lock:
            pos = self._state_pos.get(joint_name, 0.0)
            vel = self._state_vel.get(joint_name, 0.0)
            cur = self._state_eff.get(joint_name, 0.0)
        return JointSample(
            encoder_position=pos,
            estimated_velocity=vel,
            motor_current=abs(cur),
            timestamp=time.monotonic(),
        )

    def _search_loop(self) -> None:
        jn = self._search_joint
        if jn is None:
            return
        dt = self._cfg.search_period_s
        limit = self._limits[jn]
        while not self._search_stop.is_set():
            with self._lock:
                base = self._cmd_positions.get(jn, self._state_pos.get(jn, 0.0))
                qd = self._state_vel.get(jn, 0.0)
            if self._search_mode == "torque":
                tau_cmd = self._td_sign * self._td_tau - self._td_damp * qd
                v_cmd = self._cfg.torque2vel * tau_cmd
                vmax = self._cfg.torque_search_vel_max
                if v_cmd > vmax:
                    v_cmd = vmax
                elif v_cmd < -vmax:
                    v_cmd = -vmax
                nxt = base + v_cmd * dt
            else:
                nxt = base + self._search_vel * dt
            nxt = max(limit.lower, min(limit.upper, nxt))
            with self._lock:
                self._cmd_positions[jn] = nxt
            self._publish_cmd(vel_scale=0.4, time_ref=self._cfg.time_ref_search_step_s)
            time.sleep(dt)

    def start_velocity_search(self, joint_name: str, velocity: float) -> None:
        self.stop_joint(joint_name)
        if not self._debug_enabled:
            raise RuntimeError("上身调试未开启")
        self._search_mode = "velocity"
        self._search_joint = joint_name
        self._search_vel = float(velocity)
        self._search_stop.clear()
        self._search_thread = threading.Thread(target=self._search_loop, daemon=True)
        self._search_thread.start()

    def start_torque_damping_search(
        self, joint_name: str, sign: int, constant_torque: float, damping: float
    ) -> None:
        self.stop_joint(joint_name)
        if not self._debug_enabled:
            raise RuntimeError("上身调试未开启")
        self._search_mode = "torque"
        self._search_joint = joint_name
        self._td_sign = 1.0 if int(sign) >= 0 else -1.0
        self._td_tau = max(0.0, float(constant_torque))
        self._td_damp = max(0.0, float(damping))
        self._search_stop.clear()
        self._search_thread = threading.Thread(target=self._search_loop, daemon=True)
        self._search_thread.start()

    def stop_joint(self, joint_name: str) -> None:
        self._search_stop.set()
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join(timeout=2.0)
        self._search_thread = None
        # 用当前状态保持一次，避免继续滑动
        with self._lock:
            if joint_name in self._state_pos:
                self._cmd_positions[joint_name] = self._state_pos[joint_name]
                self._publish_cmd(vel_scale=0.2, time_ref=0.3)

    def apply_zero_offset(self, joint_name: str, offset: float) -> None:
        self._node.get_logger().info(
            "关节 %s 计算零偏 offset=%.6f rad (需写入驱动或从 YAML 应用)" % (joint_name, offset)
        )

    def persist_zero_offsets(self, offsets: Dict[str, float]) -> None:
        path = self._cfg.offsets_file
        write_zero_offsets_yaml(
            path,
            offsets,
            header_lines=(
                "CASBOT02 upper body — hard-stop zero offsets (radians).",
                "SetBool /motion/upper_body_debug; cmd /upper_body_debug/joint_cmd.",
            ),
        )
        self._node.get_logger().info("已写入零偏 YAML: %s" % path)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        self._search_stop.set()
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join(timeout=1.0)
        if self._debug_enabled:
            self.enable_upper_body_debug(False, timeout_s=3.0)


def _load_float_map(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        return {}
    return {str(k): float(v) for k, v in data.items()}


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="CASBOT02 上身硬限位零偏校准 (ROS2)")
    p.add_argument(
        "--urdf",
        type=Path,
        default=Path("casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf"),
    )
    p.add_argument("--arm", choices=("left", "right"), required=True)
    p.add_argument(
        "--print-plan-only", action="store_true", help="只打印计划 JSON 并退出，不连真机"
    )
    p.add_argument(
        "--joint-states",
        default="/joint_states",
        help="关节状态 topic（文档 2.5.3 默认为 /joint_states）",
    )
    p.add_argument("--encoder-signs", type=Path, help="JSON/YAML: joint_name -> +1 或 -1，缺省为全 +1")
    p.add_argument("--current-thresholds", type=Path, help="JSON/YAML: joint_name -> 判停电流/力矩阈值")
    p.add_argument("--persist", action="store_true", help="校准后写入 --offsets-out")
    p.add_argument("--offsets-out", type=Path, default=Path("zero_offsets.yaml"))
    p.add_argument(
        "--search-mode",
        choices=("torque_damping", "velocity"),
        default="torque_damping",
        help="找限位：torque_damping=恒力矩+阻尼(位置流用 torque2vel 等效)；velocity=小步进位置",
    )
    p.add_argument("--torque-nm", type=float, default=8.0, help="力矩幅值 (N·m)，与 URDF/实机量纲一致为妥")
    p.add_argument("--damping", type=float, default=3.0, help="阻尼 (N·m·s/rad)")
    p.add_argument(
        "--torque2vel",
        type=float,
        default=0.012,
        help="力矩模式：τ→等效角速度 比例 (rad/s)/(N·m)，需按实机调",
    )
    p.add_argument(
        "--torque-search-vmax",
        type=float,
        default=0.12,
        help="力矩模式：等效角速度限幅 (rad/s)",
    )
    args = p.parse_args()

    plan = build_default_arm_calibration_plan(args.urdf, args.arm)
    if args.print_plan_only:
        print(plan_summary(plan))
        return 0

    rclpy.init()
    node = Node("casbot_zero_offset_calib")

    arm_joints = arm_joint_names(args.arm)
    signs = _load_float_map(args.encoder_signs)
    for j in arm_joints:
        if j not in signs:
            signs[j] = 1.0

    if args.current_thresholds and args.current_thresholds.is_file():
        thresholds = _load_float_map(args.current_thresholds)
    else:
        thresholds = {}
    for j in arm_joints:
        if j not in thresholds:
            thresholds[j] = 5.0

    cfg = Ros2UpperBodyConfig(
        urdf_path=args.urdf,
        topic_joint_states=args.joint_states,
        offsets_file=args.offsets_out,
        torque2vel=float(args.torque2vel),
        torque_search_vel_max=float(args.torque_search_vmax),
    )
    hardware = UpperBodyDebugHardware(node, cfg, arm_joints)
    if not hardware.enable_upper_body_debug(True):
        hardware.close()
        node.destroy_node()
        rclpy.shutdown()
        return 1

    calib_cfg = HardStopCalibratorConfig(
        encoder_signs=signs,
        current_thresholds=thresholds,
        search_mode=str(args.search_mode),
        torque_search_nm=float(args.torque_nm),
        torque_damping_nm_s=float(args.damping),
    )
    calibrator = HardStopCalibrator(calib_cfg)
    try:
        hw = cast(CalibrationHardware, hardware)
        offsets = calibrator.calibrate(plan, hw, persist=args.persist)
        node.get_logger().info("校准结果: %s" % json.dumps({k: round(v, 6) for k, v in offsets.items() }))
    except Exception as exc:
        node.get_logger().error("校准失败: %s" % exc)
        raise
    finally:
        hardware.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
