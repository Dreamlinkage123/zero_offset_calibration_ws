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
import logging
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

from .hard_stop_calibration import (
    CalibrationHardware,
    HardStopCalibrator,
    HardStopCalibratorConfig,
    HardStopDetectorConfig,
    JointLimit,
    JointSample,
    arm_joint_names,
    arm_reset_waypoints,
    arm_rest_pose,
    arm_setup_waypoints,
    build_default_arm_calibration_plan,
    detect_instrument_from_urdf_path,
    normalize_plan_instrument,
    parse_joint_limits,
    plan_summary,
    write_joint_pos_offset_yaml,
    write_zero_offsets_yaml,
)
from ._paths import default_urdf_path


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
    move_timeout_s: float = 15.0
    # 无进度早停：在滑窗 move_stuck_window_s 秒内，若某未达标关节位置最大变化 <
    # move_stuck_epsilon，视为卡死，立即抛 TimeoutError，而不是傻等到 move_timeout_s。
    move_stuck_window_s: float = 3.0
    move_stuck_epsilon: float = 0.010
    offsets_file: Path = field(default_factory=lambda: Path("src/config/joint_pos_offset.yaml"))
    # 仅力矩+阻尼找限位：UpperJointData 为位置流，用 τ ≈ b·(q̇_ref−q̇) 的等价目标速度 (rad/s) / (N·m)
    torque2vel: float = 0.012
    torque_search_vel_max: float = 0.12
    ros_joint_naming: str = "no_joint_suffix"  # "urdf" or "no_joint_suffix"



def _build_name_maps(
    urdf_names: list[str], naming: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional maps between URDF joint names and ROS wire names.

    ``naming="no_joint_suffix"`` strips the ``_joint`` suffix for the wire.
    ``naming="urdf"`` uses the URDF name unchanged.
    """
    urdf_to_ros: dict[str, str] = {}
    ros_to_urdf: dict[str, str] = {}
    for urdf_name in urdf_names:
        if naming == "no_joint_suffix" and urdf_name.endswith("_joint"):
            ros_name = urdf_name[: -len("_joint")]
        else:
            ros_name = urdf_name
        urdf_to_ros[urdf_name] = ros_name
        ros_to_urdf[ros_name] = urdf_name
    return urdf_to_ros, ros_to_urdf


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

        all_urdf = list(UPPER_JOINTS_DOC_ORDER)
        self._urdf_to_ros, self._ros_to_urdf = _build_name_maps(
            all_urdf, cfg.ros_joint_naming,
        )

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
                urdf_name = self._ros_to_urdf.get(name, name)
                if i < len(msg.position):
                    self._state_pos[urdf_name] = float(msg.position[i])
                if i < len(msg.velocity):
                    self._state_vel[urdf_name] = float(msg.velocity[i])
                else:
                    self._state_vel[urdf_name] = 0.0
                if i < len(msg.effort):
                    self._state_eff[urdf_name] = float(msg.effort[i])
                else:
                    self._state_eff[urdf_name] = 0.0
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
        for urdf_name in UPPER_JOINTS_DOC_ORDER:
            if urdf_name in self._cmd_positions:
                pos = self._cmd_positions[urdf_name]
            elif urdf_name in self._state_pos:
                pos = self._state_pos[urdf_name]
            else:
                continue
            ros_name = self._urdf_to_ros.get(urdf_name, urdf_name)
            js.name.append(ros_name)
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
        # 等待进入容差；同时检测「无进度卡死」以早停
        t0 = time.monotonic()
        # 关节位置历史：{joint: [(t, pos), ...]}，窗口长度 move_stuck_window_s
        history: Dict[str, list] = {j: [] for j in pose}
        while time.monotonic() - t0 < self._cfg.move_timeout_s:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            now = time.monotonic()
            with self._lock:
                snapshot = {j: self._state_pos.get(j, 0.0) for j in pose}
            ok = all(
                abs(snapshot[j] - float(pose[j])) < self._cfg.move_tolerance_rad for j in pose
            )
            if ok:
                return
            # 更新各关节位置历史并在窗口满时做卡死检测
            stuck: List[str] = []
            window = self._cfg.move_stuck_window_s
            eps = self._cfg.move_stuck_epsilon
            for j in pose:
                hist = history[j]
                hist.append((now, snapshot[j]))
                while hist and now - hist[0][0] > window:
                    hist.pop(0)
                if hist and hist[-1][0] - hist[0][0] >= window * 0.95:
                    span = max(p for _, p in hist) - min(p for _, p in hist)
                    not_converged = abs(snapshot[j] - float(pose[j])) >= self._cfg.move_tolerance_rad
                    if not_converged and span < eps:
                        stuck.append(j)
            if stuck and window > 0.0:
                stuck_details = "; ".join(
                    "%s: cur=%.3f tgt=%.3f span=%.4f" % (
                        j, snapshot[j], float(pose[j]),
                        max(p for _, p in history[j]) - min(p for _, p in history[j]),
                    )
                    for j in stuck
                )
                self._node.get_logger().warn(
                    "move_to_pose 检出卡死关节 (无进度 %.1fs, ε=%.3f): %s" % (
                        window, eps, stuck_details,
                    )
                )
                raise TimeoutError(
                    "move_to_pose 关节卡死，无进度 %.1fs (tol=%.3f, ε=%.3f): %s" % (
                        window, self._cfg.move_tolerance_rad, eps, stuck_details,
                    )
                )
        with self._lock:
            details = "; ".join(
                "%s: cur=%.3f tgt=%.3f" % (
                    j, self._state_pos.get(j, float("nan")), float(pose[j]),
                )
                for j in pose
            )
        self._node.get_logger().warn(
            "move_to_pose 等待收敛 (tol=%.3f): %s" % (self._cfg.move_tolerance_rad, details)
        )
        raise TimeoutError("move_to_pose 超时 (tol=%.3f): %s" % (self._cfg.move_tolerance_rad, details))

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

    _HLMOTION_OFFSET_PATH = Path("/workspace/hl_motion/hl_config/joint_pos_offset.yaml")

    def persist_zero_offsets(self, offsets: Dict[str, float]) -> None:
        has_left = any(k.startswith("left_") for k in offsets)
        has_right = any(k.startswith("right_") for k in offsets)
        if has_left and has_right:
            arm = "both"
        elif has_left:
            arm = "left"
        else:
            arm = "right"
        header = (
            "CASBOT02 upper body — hard-stop zero offsets (radians).",
            "SetBool /motion/upper_body_debug; cmd /upper_body_debug/joint_cmd.",
        )
        path = self._cfg.offsets_file
        write_joint_pos_offset_yaml(path, offsets, arm, header_lines=header)
        self._node.get_logger().info("已写入零偏 YAML: %s" % path)
        try:
            write_joint_pos_offset_yaml(
                self._HLMOTION_OFFSET_PATH, offsets, arm, header_lines=header,
            )
            self._node.get_logger().info(
                "已同步零偏 YAML: %s" % self._HLMOTION_OFFSET_PATH
            )
        except OSError as exc:
            self._node.get_logger().warn(
                "同步到 %s 失败（权限或路径不存在）: %s"
                % (self._HLMOTION_OFFSET_PATH, exc)
            )

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


def _run_setup_waypoints(
    hardware: "UpperBodyDebugHardware",
    plan,
    arm: str,
    instrument: str,
    speed_scale: float,
) -> None:
    """与 :func:`mujoco_hard_stop_calibration._set_neutral_and_settle` 等价的真机预备序列。

    顺序：
      1. 先去 :func:`arm_rest_pose` 给出的下垂外展 10° rest 姿态，作为已知安全起点；
      2. 依次执行 :func:`arm_setup_waypoints` 返回的 waypoint；
         - 有乐器（``|target_pitch|>1.5``）时为 [WP1 展 roll, WP2 抬 pitch, neutral]；
         - 无乐器时直接 [neutral]。

    与仿真不同之处：仿真会同时把另一臂/腰/头压到已知 base，真机我们只下发本臂的关节
    指令（其余关节保持 ``UpperBodyDebugHardware`` 启动时种子的当前值），避免越权
    干涉操作员手动摆好的姿态。

    每一步使用与中性姿态保持相同的 ``speed_scale``（默认 0.15，与 ``HardStopCalibratorConfig.pose_speed_scale`` 一致），保证慢速可控。
    """
    logger = hardware._node.get_logger()
    rest = arm_rest_pose(arm)
    waypoints = arm_setup_waypoints(arm, plan.neutral_pose, instrument=instrument)
    label = "乐器(%s)" % instrument if instrument else "裸机"
    logger.info(
        "[%s] 预备 waypoint：rest → %d 步 → neutral（模式: %s）"
        % (arm, len(waypoints), label)
    )

    logger.info("[%s] 进入 rest pose（双臂下垂外展 10°）" % arm)
    hardware.move_to_pose(rest, speed_scale=speed_scale)

    for i, wp in enumerate(waypoints):
        is_last = i == len(waypoints) - 1
        tag = "neutral" if is_last else "WP%d" % (i + 1)
        logger.info("[%s] 进入 %s" % (arm, tag))
        hardware.move_to_pose(wp, speed_scale=speed_scale)


def _run_reset_waypoints(
    hardware: "UpperBodyDebugHardware",
    plan,
    arm: str,
    instrument: str,
    speed_scale: float,
) -> None:
    """与 :func:`mujoco_hard_stop_calibration._reset_arm` 等价的真机收尾序列。

    依次执行 :func:`arm_reset_waypoints` 返回的 [neutral, (rev-WP2,) rest]。任何一步
    抛错都仅记录警告并继续后续 waypoint，避免把手臂卡在高抬臂姿态。
    """
    logger = hardware._node.get_logger()
    waypoints = arm_reset_waypoints(arm, plan.neutral_pose, instrument=instrument)
    logger.info("[%s] 收尾 waypoint：neutral → %d 步 → rest" % (arm, len(waypoints)))
    for i, wp in enumerate(waypoints):
        is_last = i == len(waypoints) - 1
        tag = "rest" if is_last else (
            "neutral" if i == 0 else "rev-WP%d" % i
        )
        try:
            logger.info("[%s] 退回 %s" % (arm, tag))
            hardware.move_to_pose(wp, speed_scale=speed_scale)
        except Exception as exc:  # noqa: BLE001 - 收尾尽力即可
            logger.warn("[%s] 退回 %s 失败，继续下一步: %s" % (arm, tag, exc))


def _setup_logging_bridge(node: Node) -> None:
    """将 :mod:`zero_offset_calibration.hard_stop_calibration` 的 Python logging 桥接到 ROS 2 logger。"""
    ros_logger = node.get_logger()

    class _RosBridge(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                ros_logger.error(msg)
            elif record.levelno >= logging.WARNING:
                ros_logger.warn(msg)
            else:
                ros_logger.info(msg)

    bridge = _RosBridge()
    bridge.setFormatter(logging.Formatter("%(message)s"))
    calib_logger = logging.getLogger("zero_offset_calibration.hard_stop_calibration")
    calib_logger.addHandler(bridge)
    calib_logger.setLevel(logging.DEBUG)


def main() -> int:
    """CLI：初始化 ROS 2 → 开启上身调试 → 运行 :class:`HardStopCalibrator` → 关闭节点。

    真机默认判停阈值较 MuJoCo 仿真放宽（见 ``--help``），可用 ``--skip-on-timeout``
    让个别关节超时时跳过而非中止整个标定；搜索循环每 2 s 输出一行诊断日志。
    """
    import argparse

    p = argparse.ArgumentParser(
        description="CASBOT02 上身硬限位零偏校准 (ROS2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="URDF 路径；未指定时根据 --instrument 自动选择对应 URDF",
    )
    p.add_argument("--arm", choices=("left", "right"), required=True)
    p.add_argument(
        "--instrument",
        choices=("auto", "none", "bass", "guitar", "keyboard"),
        default="bass",
        help="乐器配置（同时决定默认 URDF 和 neutral/hold pose 避障路径）："
             "bass(默认)=_bass.urdf + bass 避障姿态；guitar=_guitar.urdf + guitar 避障姿态；"
             "auto=按 --urdf 文件名后缀自动识别；none=裸机姿态（与仿真 instrument='' 一致）；"
             "keyboard=电子琴在身前，使用 guitar 高抬臂轨迹但读 bass URDF。"
             "需与真实穿戴的乐器一致，否则会与琴体发生碰撞。",
    )
    p.add_argument(
        "--pose-speed-scale",
        type=float,
        default=0.15,
        help="预备/收尾 waypoint 的发布 vel_scale（默认 0.15，与 HardStopCalibratorConfig.pose_speed_scale 一致）",
    )
    p.add_argument(
        "--skip-setup-waypoints",
        action="store_true",
        help="跳过 rest→WP1→WP2→neutral 预备轨迹（不推荐：仅当外部已经把手臂摆到 neutral 时使用）",
    )
    p.add_argument(
        "--skip-reset-waypoints",
        action="store_true",
        help="跳过 neutral→rev-WP2→rest 收尾轨迹（不推荐：只在调试中需要保留最后姿态时使用）",
    )
    p.add_argument(
        "--print-plan-only", action="store_true", help="只打印计划 JSON 并退出，不连真机"
    )
    p.add_argument(
        "--joint-states",
        default="/joint_states",
        help="关节状态 topic（文档 2.5.3 默认为 /joint_states）",
    )
    p.add_argument(
        "--ros-joint-naming",
        choices=("urdf", "no_joint_suffix"),
        default="no_joint_suffix",
        help="真机关节名映射：no_joint_suffix(默认)=/joint_states 中去掉 _joint；urdf=与 URDF 名一致",
    )
    p.add_argument(
        "--move-tolerance",
        type=float,
        default=0.05,
        help="move_to_pose 到位容差 (rad)，默认 0.05",
    )
    p.add_argument(
        "--move-timeout",
        type=float,
        default=15.0,
        help="move_to_pose 最长等待时间 (s)，默认 15.0；实际通常靠卡死早停先触发",
    )
    p.add_argument(
        "--move-stuck-window",
        type=float,
        default=3.0,
        help="move_to_pose 无进度检测窗口 (s)；窗口内位置最大变化<epsilon 且未达标即早停（默认 3.0，<=0 关闭）",
    )
    p.add_argument(
        "--move-stuck-epsilon",
        type=float,
        default=0.010,
        help="move_to_pose 无进度检测位置变化门限 (rad)，默认 0.010",
    )
    p.add_argument("--encoder-signs", type=Path, help="JSON/YAML: joint_name -> +1 或 -1，缺省为全 +1")
    p.add_argument("--current-thresholds", type=Path, help="JSON/YAML: joint_name -> 判停电流/力矩阈值")
    p.add_argument(
        "--default-current-threshold",
        type=float,
        default=5.0,
        help="未在 --current-thresholds 中列出的非腕关节使用此值 (默认 5.0)",
    )
    p.add_argument(
        "--wrist-current-threshold",
        type=float,
        default=0.0,
        help="腕关节的判停电流阈值 (默认 0.0 即禁用电流判据，仅靠位置+速度+时间判停)",
    )
    p.add_argument("--persist", action="store_true", help="校准后写入 --offsets-out")
    p.add_argument("--offsets-out", type=Path, default=Path("src/config/joint_pos_offset.yaml"))
    p.add_argument(
        "--search-mode",
        choices=("torque_damping", "velocity"),
        default="torque_damping",
        help="找限位：torque_damping=恒力矩+阻尼(位置流用 torque2vel 等效)；velocity=小步进位置",
    )
    p.add_argument("--torque-nm", type=float, default=8.0, help="力矩幅值 (N·m)")
    p.add_argument("--damping", type=float, default=3.0, help="阻尼 (N·m·s/rad)")
    p.add_argument(
        "--torque2vel",
        type=float,
        default=0.012,
        help="力矩 → 等效角速度 比例 (rad/s)/(N·m)",
    )
    p.add_argument(
        "--torque-search-vmax",
        type=float,
        default=0.12,
        help="力矩模式等效角速度限幅 (rad/s)",
    )

    det_g = p.add_argument_group("判停检测器 (HardStopDetector) — 真机默认比仿真宽松")
    det_g.add_argument("--velocity-epsilon", type=float, default=0.08,
                       help="|ω| 低于此视为停住 (rad/s)，默认 0.08（仿真用 0.015）")
    det_g.add_argument("--position-epsilon", type=float, default=0.012,
                       help="滑窗内位置峰峰值阈值 (rad)，默认 0.012（仿真用 0.003）")
    det_g.add_argument("--stall-time", type=float, default=0.50,
                       help="判停所需滑窗最短持续时间 (s)，默认 0.50（仿真用 0.20）")
    det_g.add_argument("--search-timeout", type=float, default=20.0,
                       help="单关节搜索超时 (s)，默认 20.0（仿真用 10.0）。"
                            "若启用 --stuck-abort-seconds，通常卡滞会提前触发而非等到此超时")
    det_g.add_argument("--stuck-abort-seconds", type=float, default=4.0,
                       help="卡滞早停：除几何判据外全部满足且离 stop_angle 超标持续该秒数时"
                            "立即终止搜索（默认 4.0，≤0 关闭）")
    det_g.add_argument("--min-current-ratio", type=float, default=0.10,
                       help="|effort|>=threshold×ratio 视为电流达标，默认 0.10（仿真用 0.30）")
    det_g.add_argument("--backoff-seconds", type=float, default=0.20,
                       help="检出硬限位后暂停时间 (s)，默认 0.20")
    det_g.add_argument("--min-search-travel", type=float, default=0.08,
                       help="要求关节自搜索开始起实际移动 ≥ 该距离 (rad) 才允许判停，"
                            "防止在 approach 点直接误判（默认 0.08，仿真/禁用用 0.0）")
    det_g.add_argument("--max-expected-offset", type=float, default=0.60,
                       help="几何合理性门限：判停位置必须在 stop_angle 的 ±该距离 (rad) "
                            "邻域内才接受，避免关节在远离限位处被误判卡停"
                            "（默认 0.60，对手腕等机械回差较大的关节再调大；≤0 关闭）")
    det_g.add_argument("--effort-baseline-seconds", type=float, default=0.30,
                       help="动态 effort 基线窗口：搜索前该秒数内取 |effort| 最大值为自由"
                            "运动基线（默认 0.30）")
    det_g.add_argument("--effort-rise-nm", type=float, default=0.15,
                       help="动态 effort 增量门限 (Nm)：要求 |effort|≥基线+该值才允许判停，"
                            "对电流绝对门限不可靠的小电机关节尤其有效（默认 0.15，≤0 关闭）")
    det_g.add_argument("--skip-on-timeout", action="store_true",
                       help="某关节超时时跳过继续下一个，而非中止整个标定")

    args = p.parse_args()

    _INSTRUMENT_URDF = {
        "bass": "CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf",
        "guitar": "CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf",
    }
    if args.urdf is None:
        inst_key = args.instrument if args.instrument in _INSTRUMENT_URDF else "bass"
        args.urdf = default_urdf_path(_INSTRUMENT_URDF[inst_key])

    if args.instrument == "auto":
        detected = detect_instrument_from_urdf_path(args.urdf)
        instrument_label = detected or "none"
        plan_instrument = normalize_plan_instrument(detected)
    elif args.instrument == "none":
        instrument_label = "none"
        plan_instrument = ""
    else:
        instrument_label = str(args.instrument)
        plan_instrument = normalize_plan_instrument(instrument_label)

    plan = build_default_arm_calibration_plan(
        args.urdf, args.arm, instrument=plan_instrument
    )
    if args.print_plan_only:
        print(plan_summary(plan))
        return 0

    rclpy.init()
    node = Node("casbot_zero_offset_calib")
    _setup_logging_bridge(node)

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
            if "wrist" in j:
                thresholds[j] = float(args.wrist_current_threshold)
            else:
                thresholds[j] = float(args.default_current_threshold)

    cfg = Ros2UpperBodyConfig(
        urdf_path=args.urdf,
        topic_joint_states=args.joint_states,
        ros_joint_naming=str(args.ros_joint_naming),
        move_tolerance_rad=float(args.move_tolerance),
        move_timeout_s=float(args.move_timeout),
        move_stuck_window_s=float(args.move_stuck_window),
        move_stuck_epsilon=float(args.move_stuck_epsilon),
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

    detector = HardStopDetectorConfig(
        velocity_epsilon=float(args.velocity_epsilon),
        position_window_epsilon=float(args.position_epsilon),
        stall_time_seconds=float(args.stall_time),
        sample_timeout_seconds=float(args.search_timeout),
        min_current_ratio=float(args.min_current_ratio),
        backoff_seconds=float(args.backoff_seconds),
        min_search_travel=float(args.min_search_travel),
        max_expected_offset=float(args.max_expected_offset),
        effort_baseline_seconds=float(args.effort_baseline_seconds),
        effort_rise_nm=float(args.effort_rise_nm),
        stuck_abort_seconds=float(args.stuck_abort_seconds),
    )
    calib_cfg = HardStopCalibratorConfig(
        encoder_signs=signs,
        current_thresholds=thresholds,
        detector=detector,
        search_mode=str(args.search_mode),
        torque_search_nm=float(args.torque_nm),
        torque_damping_nm_s=float(args.damping),
    )

    node.get_logger().info("关节命名: %s (urdf->ros 映射 %d 条)" % (
        cfg.ros_joint_naming,
        sum(1 for u, r in _build_name_maps(list(UPPER_JOINTS_DOC_ORDER), cfg.ros_joint_naming)[0].items() if u != r),
    ))
    node.get_logger().info(
        "乐器: input=%s -> plan_instrument=%r （空字符串为裸机；guitar/bass 走高抬臂避障姿态）"
        % (instrument_label, plan_instrument)
    )
    node.get_logger().info(
        "判停参数: vel_eps=%.4f pos_eps=%.4f stall=%.2fs timeout=%.0fs "
        "cur_ratio=%.2f default_thr=%.2f min_travel=%.3f "
        "max_offset=%.3f eff_base=%.2fs eff_rise=%.3f stuck_abort=%.1fs "
        "skip_on_timeout=%s" % (
            detector.velocity_epsilon, detector.position_window_epsilon,
            detector.stall_time_seconds, detector.sample_timeout_seconds,
            detector.min_current_ratio,
            float(args.default_current_threshold),
            detector.min_search_travel,
            detector.max_expected_offset,
            detector.effort_baseline_seconds,
            detector.effort_rise_nm,
            detector.stuck_abort_seconds,
            bool(args.skip_on_timeout),
        )
    )

    calibrator = HardStopCalibrator(calib_cfg)
    pose_speed = float(args.pose_speed_scale)
    try:
        hw = cast(CalibrationHardware, hardware)
        if not args.skip_setup_waypoints:
            _run_setup_waypoints(hardware, plan, args.arm, plan_instrument, pose_speed)
        else:
            node.get_logger().warn(
                "已通过 --skip-setup-waypoints 跳过 rest/WP/neutral 预备轨迹；"
                "请确保手臂已经在 neutral 附近，否则可能与乐器碰撞"
            )
        offsets = calibrator.calibrate(
            plan, hw,
            persist=args.persist,
            skip_on_timeout=bool(args.skip_on_timeout),
        )
        node.get_logger().info(
            "校准结果: %s" % json.dumps({k: round(v, 6) for k, v in offsets.items()})
        )
    except Exception as exc:
        node.get_logger().error("校准失败: %s" % exc)
        raise
    finally:
        try:
            if not args.skip_reset_waypoints:
                _run_reset_waypoints(hardware, plan, args.arm, plan_instrument, pose_speed)
        except Exception as exc:  # noqa: BLE001 - 收尾失败不能再抛
            node.get_logger().warn("收尾 waypoint 出现异常（已忽略以便正常关闭）: %s" % exc)
        hardware.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
