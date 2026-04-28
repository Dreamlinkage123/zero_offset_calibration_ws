#!/usr/bin/env python3
"""Hard-stop based joint zero-offset calibration planner and runtime.

架构
----
本模块分两层，便于在「无实机」环境下规划标定序列，在「有实机/仿真」时通过适配器执行：

1. **规划层（URDF）**：从 URDF 读取关节限位，为左/右 7-DOF 手臂生成每关节一步的
   :class:`CalibrationStep`（接近角、搜索方向、保持姿态等）。
2. **运行时骨架**：:class:`HardStopCalibrator` 按 :class:`CalibrationPlan` 顺序调用
   :class:`CalibrationHardware` 协议（运动、采样、找限位、写零偏）。

零偏定义与计算（与 README 一致）
-------------------------------
记关节在控制器中的读数为 ``encoder_reading``，编码器方向为 ``encoder_sign ∈ {+1,-1}``，
URDF/模型给出的硬限位参考角为 ``θ_stop``（本规划中为 ``CalibrationStep.stop_angle``，
即沿选定方向顶到限位时「关节角真值」应等于该边界）。

在硬限位处采样读数 ``θ_enc`` 后，零偏定义为::

    offset = θ_stop − encoder_sign × θ_enc

标定后使用时：::

    joint_angle = encoder_sign × encoder_reading + offset

这样当机械再次顶到同一硬限时，``joint_angle ≈ θ_stop``。

单步标定流程（概念）
-------------------
对每一步：先 ``move_to_pose(hold_pose)`` 把非目标关节摆到安全姿态；再沿
``search_velocity`` 方向低速顶限位；用 :class:`HardStopDetectorConfig` 判定「堵转」
（位置几乎不动、速度近零、电流足够）；停机、回退 ``backoff_angle``、计算并
``apply_zero_offset``。

仓库内仅含模型资产，协议方法由 ``mujoco_hard_stop_calibration`` 或实机适配器实现。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol

logger = logging.getLogger(__name__)


def write_zero_offsets_yaml(
    path: str | Path,
    offsets: Mapping[str, float],
    *,
    header_lines: tuple[str, ...] = (
        "Joint zero-offset calibration result (radians).",
        "joint_angle = encoder_sign * encoder_reading + offset",
    ),
) -> None:
    """将零偏字典写入 YAML 风格文本文件（不依赖 PyYAML）。

    仅当键名不符合 ``[A-Za-z0-9_]+`` 时用 JSON 引号包裹，避免非法 YAML 键。
    """

    path = Path(path)
    lines: List[str] = []
    for h in header_lines:
        lines.append("# " + h)
    if header_lines:
        lines.append("")
    for key in sorted(offsets.keys()):
        k = str(key)
        if re.fullmatch(r"[A-Za-z0-9_]+", k):
            key_out = k
        else:
            key_out = json.dumps(k, ensure_ascii=False)
        val = float(offsets[key])
        lines.append(f"{key_out}: {val!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# YAML 数组输出时的关节顺序（与用户约定一致，与标定规划顺序无关）
_LEFT_ARM_YAML_ORDER = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
_RIGHT_ARM_YAML_ORDER = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

_DEFAULT_JOINT_POS_OFFSET = {
    "left_leg_pos": [0.0] * 6,
    "right_leg_pos": [0.0] * 6,
    "left_arm_pos": [0.0] * 7,
    "right_arm_pos": [0.0] * 7,
    "head_pos": [0.0, 0.0],
    "waist_pos": [0.0],
}

_YAML_KEY_ORDER = [
    "left_leg_pos", "right_leg_pos",
    "left_arm_pos", "right_arm_pos",
    "head_pos", "waist_pos",
]


def _format_array(arr: List[float]) -> str:
    """将浮点数组格式化为 YAML 行内数组字符串。"""
    return "[" + ", ".join(f"{v:.10f}" if v != 0.0 else "0.0" for v in arr) + "]"


def _parse_joint_pos_offset_yaml(path: Path) -> Dict[str, List[float]]:
    """读取已有的 joint_pos_offset YAML，返回各 key 对应的数组。

    不依赖 PyYAML；仅识别本函数写出的固定格式。
    """
    data = dict(_DEFAULT_JOINT_POS_OFFSET)
    if not path.is_file():
        return {k: list(v) for k, v in data.items()}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("joint_pos_offset"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if key in data and val.startswith("["):
            val = val.strip("[] ")
            if val:
                data[key] = [float(x) for x in val.split(",")]
    return {k: list(v) for k, v in data.items()}


def write_joint_pos_offset_yaml(
    path: str | Path,
    offsets: Mapping[str, float],
    arm: str,
    *,
    header_lines: tuple[str, ...] = (
        "CASBOT02 joint position offset (radians).",
    ),
) -> None:
    """将零偏以 ``joint_pos_offset`` 数组格式写入 YAML（不依赖 PyYAML）。

    只更新 ``arm`` 对应的数组（left/right），保留文件中其余字段不变。
    ``arm="both"`` 同时更新两臂。
    """
    path = Path(path)
    data = _parse_joint_pos_offset_yaml(path)

    arms_to_update: List[str] = []
    if arm in ("left", "both"):
        arms_to_update.append("left")
    if arm in ("right", "both"):
        arms_to_update.append("right")

    for side in arms_to_update:
        order = _LEFT_ARM_YAML_ORDER if side == "left" else _RIGHT_ARM_YAML_ORDER
        arr = [offsets.get(jn, 0.0) for jn in order]
        data[f"{side}_arm_pos"] = arr

    lines: List[str] = []
    for h in header_lines:
        lines.append("# " + h)
    if header_lines:
        lines.append("")
    lines.append("joint_pos_offset:")
    for key in _YAML_KEY_ORDER:
        lines.append(f"    {key}: {_format_array(data[key])}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 规划顺序：从肩到腕，与典型装配/调试顺序一致；每关节一步。
LEFT_ARM_JOINTS = (
    "left_shoulder_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)

RIGHT_ARM_JOINTS = (
    "right_shoulder_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


# 双臂自然下垂会与大腿干涉，rest pose 让 shoulder_roll 外展 10°（≈0.175 rad）
REST_ROLL_RAD = 0.175

# guitar/bass 走"乐器感知"的高抬臂避障姿态；keyboard 在 MuJoCo 中可剥离琴体 body，
# 但实机上电子琴位于身前，仍按 guitar 的高抬臂轨迹规划（normalize_plan_instrument）。
_CONTACT_INSTRUMENTS = frozenset({"guitar", "bass"})
_PLAN_INSTRUMENT_ALIASES: Dict[str, str] = {"keyboard": "guitar"}


def normalize_plan_instrument(instrument: str) -> str:
    """把任意 ``instrument`` 标识归一到 :func:`build_default_arm_calibration_plan`
    支持的取值（``"guitar"`` / ``"bass"`` / ``""``）。

    与 :mod:`mujoco_hard_stop_calibration` 中的别名表保持一致：``"keyboard"``
    → ``"guitar"``（高抬臂轨迹）；其它非接触类乐器一律视为空。
    """
    s = (instrument or "").strip().lower()
    if not s:
        return ""
    if s in _PLAN_INSTRUMENT_ALIASES:
        return _PLAN_INSTRUMENT_ALIASES[s]
    return s if s in _CONTACT_INSTRUMENTS else ""


def _instrument_suffix_from_resource_path(path: str | Path) -> str:
    """从资源文件名（任意扩展名）的 stem 推断乐器：``*_bass`` / ``*_guitar`` / ``*_keyboard``。

    与真机/仿真统一约定；匹配不到时返回 ``""``。
    """
    stem = Path(str(path)).stem.lower()
    for suffix in ("bass", "guitar", "keyboard"):
        if stem.endswith("_" + suffix):
            return suffix
    return ""


def detect_instrument_from_urdf_path(urdf_path: str | Path) -> str:
    """从 URDF 文件名 stem 推断乐器：``*_bass.urdf`` / ``*_guitar.urdf`` / ``*_keyboard.urdf``。

    匹配不到时返回 ``""``（视为无乐器，使用 bare 标定姿态）。
    """
    return _instrument_suffix_from_resource_path(urdf_path)


def detect_instrument_from_xml_path(xml_path: str | Path) -> str:
    """从 MuJoCo ``*.xml`` 文件名 stem 推断乐器，规则与 :func:`detect_instrument_from_urdf_path` 相同。

    例如 ``..._P1L_bass.xml`` → ``"bass"``，``..._P1L.xml``（无后缀）→ ``""``。
    与仿真侧 :mod:`mujoco_hard_stop_calibration` 的碰撞过滤、标定计划乐器参数一致。
    """
    return _instrument_suffix_from_resource_path(xml_path)


def arm_rest_pose(arm: str) -> Dict[str, float]:
    """单臂 rest 姿态：双臂下垂但 ``shoulder_roll`` 外展 ``REST_ROLL_RAD`` 避开大腿。

    与 :mod:`mujoco_hard_stop_calibration._arm_rest_pose` 完全一致。
    """
    sign = 1.0 if arm == "left" else -1.0
    rest = {j: 0.0 for j in arm_joint_names(arm)}
    rest[f"{arm}_shoulder_roll_joint"] = sign * REST_ROLL_RAD
    return rest


def arm_setup_waypoints(
    arm: str,
    neutral_pose: Mapping[str, float],
    instrument: str = "",
) -> List[Dict[str, float]]:
    """从 rest 安全过渡到 ``neutral_pose`` 的 waypoint 序列（不含起点 rest）。

    与 :mod:`mujoco_hard_stop_calibration._set_neutral_and_settle` 的"先展后抬"
    保持一致：当目标 ``shoulder_pitch`` 较大（``|pitch| > 1.5`` rad，典型为
    乐器感知的高抬臂姿态）时，插入两个中间点避免前臂在抬升过程中扫过乐器：

      * WP1 — 仅把 ``shoulder_roll`` 展到目标值，``pitch``/``elbow`` 维持 0；
      * WP2 — 抬 ``pitch`` 到目标值，``elbow`` 仍保持 0；
      * 终点 — 完整 ``neutral_pose``（含 ``elbow`` 弯曲等）。

    无乐器（或 ``|pitch| ≤ 1.5``）时直接返回单元素列表 ``[neutral_pose]``。
    返回的每个 dict 仅包含本臂 7 个关节，便于上层适配器只下发必要关节。

    ``instrument`` 当前未直接影响 waypoint 拓扑（拓扑取决于 ``neutral_pose``
    本身的形态），保留为参数以便日后按乐器细化轨迹。
    """
    _ = instrument
    sp_key = f"{arm}_shoulder_pitch_joint"
    sr_key = f"{arm}_shoulder_roll_joint"
    ep_key = f"{arm}_elbow_pitch_joint"
    target_pitch = float(neutral_pose.get(sp_key, 0.0))
    target_roll = float(neutral_pose.get(sr_key, 0.0))
    needs_waypoint = abs(target_pitch) > 1.5

    arm_joints = arm_joint_names(arm)
    final_pose = {j: float(neutral_pose.get(j, 0.0)) for j in arm_joints}
    waypoints: List[Dict[str, float]] = []

    if needs_waypoint:
        wp1 = dict(final_pose)
        wp1[sp_key] = 0.0
        wp1[ep_key] = 0.0
        wp1[sr_key] = target_roll
        waypoints.append(wp1)

        wp2 = dict(final_pose)
        wp2[ep_key] = 0.0
        waypoints.append(wp2)

    waypoints.append(final_pose)
    return waypoints


def arm_reset_waypoints(
    arm: str,
    neutral_pose: Mapping[str, float],
    instrument: str = "",
) -> List[Dict[str, float]]:
    """从任意 hold/neutral 附近退回 rest 的逆序 waypoint 序列（含终点 rest）。

    与 :mod:`mujoco_hard_stop_calibration._reset_arm` 一致：起点假定为最后一步
    ``hold_pose``（与 ``neutral_pose`` 接近），先回到 ``neutral_pose`` 作为已知
    安全点；当 ``|pitch_neutral| > 1.5`` 时再走 rev-WP2（``pitch=0``，``roll`` 维持
    外展）→ rest（``roll`` 收回），避免直接从高抬臂回 rest 时前臂扫过琴身。

    无乐器（或低抬臂）时序列为 ``[neutral_pose, rest]``。
    """
    _ = instrument
    sp_key = f"{arm}_shoulder_pitch_joint"
    sr_key = f"{arm}_shoulder_roll_joint"
    target_pitch = float(neutral_pose.get(sp_key, 0.0))
    target_roll = float(neutral_pose.get(sr_key, 0.0))
    needs_waypoint = abs(target_pitch) > 1.5

    arm_joints = arm_joint_names(arm)
    rest = arm_rest_pose(arm)
    waypoints: List[Dict[str, float]] = []
    waypoints.append({j: float(neutral_pose.get(j, 0.0)) for j in arm_joints})

    if needs_waypoint:
        wp = dict(rest)
        wp[sr_key] = target_roll
        waypoints.append(wp)

    waypoints.append(dict(rest))
    return waypoints


def rad_to_deg(value: float) -> float:
    return value * 180.0 / 3.141592653589793


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


@dataclass(frozen=True)
class JointLimit:
    """单个转动关节在 URDF 中的限位与轴信息。"""

    name: str
    axis: tuple[float, float, float]
    lower: float
    upper: float
    effort: float
    velocity: float

    @property
    def center(self) -> float:
        return (self.lower + self.upper) * 0.5


@dataclass(frozen=True)
class CalibrationStep:
    """单关节硬限位标定一步：目标关节、限位侧、角度与保持姿态。"""

    target_joint: str
    stop_side: str
    stop_angle: float
    approach_angle: float
    search_velocity: float
    backoff_angle: float
    hold_pose: Dict[str, float]
    notes: str

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["stop_angle_deg"] = round(rad_to_deg(self.stop_angle), 3)
        data["approach_angle_deg"] = round(rad_to_deg(self.approach_angle), 3)
        data["search_velocity_deg_s"] = round(rad_to_deg(self.search_velocity), 3)
        data["backoff_angle_deg"] = round(rad_to_deg(self.backoff_angle), 3)
        data["hold_pose_deg"] = {
            joint: round(rad_to_deg(angle), 3)
            for joint, angle in self.hold_pose.items()
        }
        return data


@dataclass(frozen=True)
class CalibrationPlan:
    """单侧手臂完整标定计划：中立姿 + 按序各关节步骤。"""

    arm: str
    urdf_path: str
    joint_limits: Dict[str, JointLimit]
    neutral_pose: Dict[str, float]
    steps: List[CalibrationStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "arm": self.arm,
            "urdf_path": self.urdf_path,
            "neutral_pose_rad": self.neutral_pose,
            "neutral_pose_deg": {
                joint: round(rad_to_deg(angle), 3)
                for joint, angle in self.neutral_pose.items()
            },
            "joint_limits": {
                name: {
                    "axis": limit.axis,
                    "lower": limit.lower,
                    "upper": limit.upper,
                    "lower_deg": round(rad_to_deg(limit.lower), 3),
                    "upper_deg": round(rad_to_deg(limit.upper), 3),
                }
                for name, limit in self.joint_limits.items()
            },
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class JointSample:
    """单时刻关节反馈：用于硬限位检测滑窗。"""

    encoder_position: float
    estimated_velocity: float
    motor_current: float
    timestamp: float


@dataclass(frozen=True)
class HardStopDetectorConfig:
    """硬限位「堵转」判据：在滑窗内位置几乎不动、速度近零、电流足够、时间够长。"""

    min_current_ratio: float = 0.30
    velocity_epsilon: float = 0.015
    position_window_epsilon: float = 0.003
    stall_time_seconds: float = 0.20
    sample_timeout_seconds: float = 10.0
    backoff_seconds: float = 0.20
    # 防止「还没推动关节就被判停」：要求关节自搜索开始起实际移动过 ≥ 该距离（rad）
    # 默认 0.08 rad，明显小于 approach_angle→stop_angle 的 0.20 rad 裕度。
    min_search_travel: float = 0.08
    # 几何合理性：判停位置必须落在 stop_angle 的 ±该距离 邻域内才接受，避免关节
    # 在远离限位的位置因摩擦/卡滞被误判。
    # 默认 0.60 rad —— 经过现场标定校准：大于 approach 0.20 rad 的安全余量，
    # 同时容下 wrist 类小电机/有较大机械回差关节的实测停位与 URDF stop_angle 偏差。
    # 设为 <=0 可关闭此检查。
    max_expected_offset: float = 0.60
    # 动态 effort 基线：在搜索前 ``effort_baseline_seconds`` 秒内取 |effort| 最大值
    # 作为「自由运动阶段」基线；判停时要求 |effort| ≥ baseline + ``effort_rise_nm``。
    # 仅在 current_threshold<=0（绝对电流门限禁用）时生效，避免与绝对门限冲突。
    # 适合电流绝对门限不可靠的关节（如小电机的 wrist）。设 ``effort_rise_nm<=0`` 关闭。
    effort_baseline_seconds: float = 0.30
    effort_rise_nm: float = 0.15
    # 「卡滞快速失败」：若除 near_limit（几何）外所有条件都满足、
    # 但位置持续离 stop_angle 超过 max_expected_offset 的时长 ≥ 该值，
    # 立即视作搜索失败（而不是白等到 sample_timeout_seconds）。≤0 关闭。
    stuck_abort_seconds: float = 4.0


@dataclass(frozen=True)
class HardStopCalibratorConfig:
    """标定运行时参数：编码器方向、电流门限、检测器、运动/搜索模式。"""

    encoder_signs: Mapping[str, float]
    current_thresholds: Mapping[str, float]
    detector: HardStopDetectorConfig = field(default_factory=HardStopDetectorConfig)
    pose_speed_scale: float = 0.15
    settle_seconds: float = 0.30
    search_mode: str = "torque_damping"
    torque_search_nm: float = 8.0
    torque_damping_nm_s: float = 3.0


class CalibrationHardware(Protocol):
    """硬件/仿真适配器必须实现的接口：运动、采样、找限位、写零偏。"""

    def move_to_pose(self, pose: Mapping[str, float], speed_scale: float) -> None: ...

    def read_sample(self, joint_name: str) -> JointSample: ...

    def start_velocity_search(self, joint_name: str, velocity: float) -> None: ...

    def start_torque_damping_search(
        self, joint_name: str, sign: int, constant_torque: float, damping: float
    ) -> None: ...

    def stop_joint(self, joint_name: str) -> None: ...

    def apply_zero_offset(self, joint_name: str, offset: float) -> None: ...

    def persist_zero_offsets(self, offsets: Mapping[str, float]) -> None: ...

    def sleep(self, seconds: float) -> None: ...


def parse_joint_limits(urdf_path: str | Path) -> Dict[str, JointLimit]:
    """解析 URDF 中所有 ``type="revolute"`` 关节的 ``<limit>`` 与 ``<axis>``。"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits: Dict[str, JointLimit] = {}

    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue

        name = joint.get("name")
        axis_node = joint.find("axis")
        limit_node = joint.find("limit")
        if not name or axis_node is None or limit_node is None:
            continue

        axis = tuple(float(value) for value in axis_node.get("xyz", "0 0 0").split())
        limits[name] = JointLimit(
            name=name,
            axis=axis,
            lower=float(limit_node.get("lower", "0")),
            upper=float(limit_node.get("upper", "0")),
            effort=float(limit_node.get("effort", "0")),
            velocity=float(limit_node.get("velocity", "0")),
        )

    return limits


def arm_joint_names(arm: str) -> tuple[str, ...]:
    if arm == "left":
        return LEFT_ARM_JOINTS
    if arm == "right":
        return RIGHT_ARM_JOINTS
    raise ValueError(f"Unsupported arm: {arm}")


def preferred_stop_side(arm: str, joint_name: str, instrument: str = "") -> int:
    """返回该关节默认去顶的限位侧：-1 表示下限端，+1 表示上限端。"""
    _ = instrument
    side_map = {
        "left_shoulder_pitch_joint": -1,
        "left_shoulder_roll_joint": +1,
        "left_shoulder_yaw_joint": +1,
        "left_elbow_pitch_joint": +1,
        "left_wrist_yaw_joint": +1,
        "left_wrist_pitch_joint": -1,
        "left_wrist_roll_joint": -1,
        "right_shoulder_pitch_joint": -1,
        "right_shoulder_roll_joint": -1,
        "right_shoulder_yaw_joint": -1,
        "right_elbow_pitch_joint": +1,
        "right_wrist_yaw_joint": -1,
        "right_wrist_pitch_joint": -1,
        "right_wrist_roll_joint": +1,
    }
    try:
        return side_map[joint_name]
    except KeyError as exc:
        raise KeyError(f"No preferred stop configured for {arm} arm joint {joint_name}") from exc


def default_neutral_pose(
    arm: str,
    joint_limits: Mapping[str, JointLimit],
    instrument: str = "",
) -> Dict[str, float]:
    outward_roll = 0.60 if arm == "left" else -0.60
    pitch = -1.10
    elbow = -0.90
    if instrument and arm == "left":
        outward_roll = 1.5
        pitch = -2.50
        elbow = -0.50
    elif instrument and arm == "right":
        outward_roll = -1.5
        pitch = -2.50
        elbow = -0.50
    pose = {
        f"{arm}_shoulder_pitch_joint": pitch,
        f"{arm}_shoulder_roll_joint": outward_roll,
        f"{arm}_shoulder_yaw_joint": 0.0,
        f"{arm}_elbow_pitch_joint": elbow,
        f"{arm}_wrist_yaw_joint": 0.0,
        f"{arm}_wrist_pitch_joint": 0.0,
        f"{arm}_wrist_roll_joint": 0.0,
    }
    return {
        joint: clamp(angle, joint_limits[joint].lower + 0.08, joint_limits[joint].upper - 0.08)
        for joint, angle in pose.items()
    }


def step_hold_pose(
    arm: str,
    target_joint: str,
    approach_angle: float,
    neutral_pose: Mapping[str, float],
    joint_limits: Mapping[str, JointLimit],
    instrument: str = "",
) -> Dict[str, float]:
    pose: Dict[str, float] = dict(neutral_pose)

    shoulder_pitch = f"{arm}_shoulder_pitch_joint"
    shoulder_roll = f"{arm}_shoulder_roll_joint"
    shoulder_yaw = f"{arm}_shoulder_yaw_joint"
    elbow_pitch = f"{arm}_elbow_pitch_joint"
    wrist_yaw = f"{arm}_wrist_yaw_joint"
    wrist_pitch = f"{arm}_wrist_pitch_joint"
    wrist_roll = f"{arm}_wrist_roll_joint"

    has_inst = bool(instrument)
    outward_roll = 0.75 if arm == "left" else -0.75
    if has_inst and arm == "left":
        outward_roll = 1.2
    elif has_inst and arm == "right":
        outward_roll = -1.2
    pose[shoulder_roll] = clamp(
        outward_roll,
        joint_limits[shoulder_roll].lower + 0.08,
        joint_limits[shoulder_roll].upper - 0.08,
    )
    pose[elbow_pitch] = clamp(
        -0.50 if has_inst else -0.85,
        joint_limits[elbow_pitch].lower + 0.08,
        joint_limits[elbow_pitch].upper - 0.08,
    )

    if target_joint == shoulder_roll:
        sp_val = -2.50 if has_inst else -1.20
        pose[shoulder_pitch] = clamp(
            sp_val,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
        ELBOW_SHOULDER_ROLL_STEP = -1.5  # rad
        pose[elbow_pitch] = clamp(
            ELBOW_SHOULDER_ROLL_STEP,
            joint_limits[elbow_pitch].lower + 0.08,
            joint_limits[elbow_pitch].upper - 0.08,
        )
    elif target_joint == shoulder_pitch:
        pose[shoulder_roll] = clamp(
            outward_roll,
            joint_limits[shoulder_roll].lower + 0.08,
            joint_limits[shoulder_roll].upper - 0.08,
        )
        ELBOW_SHOULDER_ROLL_STEP = -1.5  # rad
        pose[elbow_pitch] = clamp(
            ELBOW_SHOULDER_ROLL_STEP,
            joint_limits[elbow_pitch].lower + 0.08,
            joint_limits[elbow_pitch].upper - 0.08,
        )
    elif target_joint == shoulder_yaw:
        pose[shoulder_pitch] = clamp(
            -2.50 if has_inst else -1.00,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
    elif target_joint == elbow_pitch:
        pose[shoulder_pitch] = clamp(
            -2.50 if has_inst else -1.00,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
        pose[shoulder_yaw] = 0.0
    elif target_joint in {wrist_yaw, wrist_pitch, wrist_roll}:
        pose[shoulder_pitch] = clamp(
            -2.50 if has_inst else -0.95,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
        pose[shoulder_yaw] = 0.0
        pose[elbow_pitch] = clamp(
            -0.50 if has_inst else -1.10,
            joint_limits[elbow_pitch].lower + 0.08,
            joint_limits[elbow_pitch].upper - 0.08,
        )

    pose[target_joint] = approach_angle
    return pose


def build_default_arm_calibration_plan(
    urdf_path: str | Path,
    arm: str,
    instrument: str = "",
) -> CalibrationPlan:
    """根据 URDF 为单侧手臂生成默认 7 步硬限位标定计划。"""
    all_limits = parse_joint_limits(urdf_path)
    joint_names = arm_joint_names(arm)
    joint_limits = {name: all_limits[name] for name in joint_names}
    neutral_pose = default_neutral_pose(arm, joint_limits, instrument=instrument)
    steps: List[CalibrationStep] = []

    for joint_name in joint_names:
        limit = joint_limits[joint_name]
        stop_direction = preferred_stop_side(arm, joint_name, instrument=instrument)
        stop_angle = limit.upper if stop_direction > 0 else limit.lower
        approach_angle = clamp(
            stop_angle - stop_direction * 0.20,
            limit.lower + 0.05,
            limit.upper - 0.05,
        )
        search_velocity = stop_direction * min(0.20, max(0.05, limit.velocity * 0.10))
        backoff_angle = clamp(
            stop_angle - stop_direction * 0.06,
            limit.lower + 0.03,
            limit.upper - 0.03,
        )
        hold_pose = step_hold_pose(
            arm, joint_name, approach_angle, neutral_pose, joint_limits,
            instrument=instrument,
        )
        steps.append(
            CalibrationStep(
                target_joint=joint_name,
                stop_side="upper" if stop_direction > 0 else "lower",
                stop_angle=stop_angle,
                approach_angle=approach_angle,
                search_velocity=search_velocity,
                backoff_angle=backoff_angle,
                hold_pose=hold_pose,
                notes=(
                    "Recommended from URDF joint limits only. "
                    "Verify the chosen stop direction on hardware before enabling torque."
                ),
            )
        )

    return CalibrationPlan(
        arm=arm,
        urdf_path=str(urdf_path),
        joint_limits=joint_limits,
        neutral_pose=neutral_pose,
        steps=steps,
    )


def search_direction_sign_for_step(step: CalibrationStep) -> int:
    """与 ``step.search_velocity`` 同号的 ±1，供力矩模式找限位。"""
    s = float(step.search_velocity)
    if s > 0.0:
        return 1
    if s < 0.0:
        return -1
    return 1


class HardStopCalibrator:
    """按 :class:`CalibrationPlan` 顺序执行硬限位标定并汇总零偏。"""

    def __init__(self, config: HardStopCalibratorConfig):
        self.config = config

    def compute_zero_offset(self, joint_name: str, encoder_position: float, reference_angle: float) -> float:
        sign = self.config.encoder_signs[joint_name]
        return reference_angle - sign * encoder_position

    def _stopped_on_hard_limit(
        self,
        history: List[JointSample],
        current_threshold: float,
        search_start_pos: Optional[float] = None,
        stop_angle: Optional[float] = None,
        joint_name: Optional[str] = None,
        effort_baseline: Optional[float] = None,
    ) -> bool:
        detector = self.config.detector
        if len(history) < 2:
            return False

        newest = history[-1]
        oldest = history[0]
        duration = newest.timestamp - oldest.timestamp
        delta = abs(newest.encoder_position - oldest.encoder_position)
        current_required = current_threshold * detector.min_current_ratio
        current_ok = current_required <= 0 or abs(newest.motor_current) >= current_required
        velocity_ok = abs(newest.estimated_velocity) <= detector.velocity_epsilon
        position_ok = delta <= detector.position_window_epsilon
        stall_ok = duration >= detector.stall_time_seconds * 0.98
        travel_ok = True
        if search_start_pos is not None and detector.min_search_travel > 0.0:
            travel_ok = (
                abs(newest.encoder_position - search_start_pos) >= detector.min_search_travel
            )
        # 几何合理性：stall 位置必须在 stop_angle 的 ±max_expected_offset 邻域
        has_geo_check = stop_angle is not None and detector.max_expected_offset > 0.0
        near_limit_ok = True
        if has_geo_check:
            # 零偏 = stop - sign × enc；|零偏| ≤ max_expected_offset
            #        ⇔ |sign × enc − stop| ≤ max_expected_offset
            sign = 1.0
            if joint_name is not None:
                sign = float(self.config.encoder_signs.get(joint_name, 1.0))
            near_limit_ok = (
                abs(sign * newest.encoder_position - stop_angle)
                <= detector.max_expected_offset
            )
        # 动态 effort 基线：|effort| ≥ baseline + rise
        # 仅当「绝对电流门限被禁用」(current_threshold<=0) 时作为替代判据启用；
        # 否则会和绝对门限冲突（肩/肘等重力负载关节在自由运动期 effort 已饱和，
        # 实际撞限位不会再显著增大，导致永远判不了停）。
        # 注意：torque-damping 搜索到位后 v≈0，输出力矩回落到 torque_search_nm，
        # 反而低于自由运动阶段的基线，effort_rise 在该模式下常常不成立。
        has_eff_check = (
            detector.effort_rise_nm > 0.0
            and current_threshold <= 0.0
            and effort_baseline is not None
        )
        effort_rise_ok = True
        if has_eff_check:
            effort_rise_ok = (
                abs(newest.motor_current) >= effort_baseline + detector.effort_rise_nm
            )
        # near_limit 与 effort_rise 都是「确认确实撞到限位」的证据。两条都开时，
        # 任一条通过即可（逻辑 OR），避免在某一条物理上不成立时造成漏检。
        if has_geo_check and has_eff_check:
            confirm_ok = near_limit_ok or effort_rise_ok
        elif has_geo_check:
            confirm_ok = near_limit_ok
        elif has_eff_check:
            confirm_ok = effort_rise_ok
        else:
            confirm_ok = True
        return (
            stall_ok
            and current_ok
            and velocity_ok
            and position_ok
            and travel_ok
            and confirm_ok
        )

    def _log_search_diagnostics(
        self,
        joint_name: str,
        history: List[JointSample],
        current_threshold: float,
        step_idx: int,
        total_steps: int,
        elapsed: float,
        search_start_pos: Optional[float] = None,
        stop_angle: Optional[float] = None,
        effort_baseline: Optional[float] = None,
    ) -> None:
        """每隔几秒输出一次判停条件诊断，便于现场调参。"""
        det = self.config.detector
        if len(history) < 2:
            logger.info(
                "[%d/%d] %s search %.1fs: 采样不足 (n=%d)",
                step_idx, total_steps, joint_name, elapsed, len(history),
            )
            return
        newest = history[-1]
        oldest = history[0]
        dur = newest.timestamp - oldest.timestamp
        delta = abs(newest.encoder_position - oldest.encoder_position)
        cur_need = current_threshold * det.min_current_ratio
        cur_ok_str = (
            "SKIP" if cur_need <= 0 else ("OK" if abs(newest.motor_current) >= cur_need else "NO")
        )
        if search_start_pos is not None and det.min_search_travel > 0.0:
            travel = abs(newest.encoder_position - search_start_pos)
            travel_ok = travel >= det.min_search_travel
            travel_str = " | travel=%.4f(>%s %.4f)" % (
                travel, "OK" if travel_ok else "NO", det.min_search_travel,
            )
        else:
            travel_str = ""
        has_geo_check = stop_angle is not None and det.max_expected_offset > 0.0
        near_ok = True
        if has_geo_check:
            sign = float(self.config.encoder_signs.get(joint_name, 1.0))
            near_err = abs(sign * newest.encoder_position - stop_angle)
            near_ok = near_err <= det.max_expected_offset
            near_str = " | dist2stop=%.4f(<%s %.4f)" % (
                near_err, "OK" if near_ok else "NO", det.max_expected_offset,
            )
        else:
            near_str = ""
        # 仅在「绝对电流门限被禁用」且 effort_rise_nm>0 时 effort_rise 才实际生效
        has_eff_check = (
            det.effort_rise_nm > 0.0
            and current_threshold <= 0.0
            and effort_baseline is not None
        )
        eff_ok = True
        if has_eff_check:
            need = effort_baseline + det.effort_rise_nm
            eff_ok = abs(newest.motor_current) >= need
            rise_str = " | eff_rise |eff|=%.3f(>%s base %.3f+%.3f=%.3f)" % (
                abs(newest.motor_current),
                "OK" if eff_ok else "NO",
                effort_baseline, det.effort_rise_nm, need,
            )
        elif det.effort_rise_nm > 0.0 and current_threshold > 0.0:
            rise_str = " | eff_rise=SKIP(abs_thr>0)"
        else:
            rise_str = ""
        # near_limit / effort_rise 只要其中一条 OK 即视为「确认到位」
        if has_geo_check and has_eff_check:
            confirm_str = " | confirm=%s(geo|eff)" % ("OK" if (near_ok or eff_ok) else "NO")
        else:
            confirm_str = ""
        logger.info(
            "[%d/%d] %s search %.1fs | "
            "pos=%.4f Δpos=%.4f(<%s %.4f) | "
            "|vel|=%.4f(<%s %.4f) | "
            "|eff|=%.3f(>%s %.3f) | "
            "win=%.3fs(>%s %.3fs)%s%s%s%s | n=%d",
            step_idx, total_steps, joint_name, elapsed,
            newest.encoder_position,
            delta, "OK" if delta <= det.position_window_epsilon else "NO", det.position_window_epsilon,
            abs(newest.estimated_velocity),
            "OK" if abs(newest.estimated_velocity) <= det.velocity_epsilon else "NO",
            det.velocity_epsilon,
            abs(newest.motor_current),
            cur_ok_str, cur_need,
            dur, "OK" if dur >= det.stall_time_seconds * 0.98 else "NO", det.stall_time_seconds,
            travel_str,
            near_str,
            rise_str,
            confirm_str,
            len(history),
        )

    def calibrate(
        self,
        plan: CalibrationPlan,
        hardware: CalibrationHardware,
        persist: bool = False,
        skip_on_timeout: bool = False,
    ) -> Dict[str, float]:
        """对 ``plan.steps`` 逐关节执行：到位 → 搜索 → 判停 → 回退 → 记零偏。

        若 ``persist`` 为 True，最后调用 ``hardware.persist_zero_offsets``。

        ``skip_on_timeout=False``（默认）：超时则抛 :class:`TimeoutError`。
        ``skip_on_timeout=True``：超时关节仅打印警告、回退到 hold pose，继续下一个。
        """
        offsets: Dict[str, float] = {}
        enc_positions: Dict[str, float] = {}
        skipped: List[str] = []
        total_steps = len(plan.steps)

        for step_idx, step in enumerate(plan.steps, 1):
            jn = step.target_joint
            logger.info(
                "[%d/%d] 标定 %s (%s, stop=%.4f rad)",
                step_idx, total_steps, jn, step.stop_side, step.stop_angle,
            )
            try:
                self._run_calibration_step(
                    step, step_idx, total_steps, hardware, offsets, skipped,
                    enc_positions=enc_positions,
                    skip_on_timeout=skip_on_timeout,
                )
            except TimeoutError as exc:
                # 该步骤任何阶段（move_to_pose / 搜索超时 / 卡死早停）抛的 TimeoutError
                # 在 skip_on_timeout 模式下统一归并为「跳过该关节，继续下一个」
                logger.warning(
                    "[%d/%d] %s 步骤超时: %s", step_idx, total_steps, jn, exc,
                )
                try:
                    hardware.stop_joint(jn)
                except Exception:  # noqa: BLE001
                    pass
                if not skip_on_timeout:
                    raise
                if jn not in skipped:
                    skipped.append(jn)
                # 尽力回撤到 hold_pose 方便下一步起步；失败也只是警告，不阻断整臂标定
                try:
                    hardware.move_to_pose(
                        step.hold_pose, speed_scale=self.config.pose_speed_scale
                    )
                    hardware.sleep(self.config.settle_seconds)
                except Exception as release_exc:  # noqa: BLE001
                    logger.warning(
                        "[%d/%d] %s 跳过后回撤失败，直接进入下一关节: %s",
                        step_idx, total_steps, jn, release_exc,
                    )

        if skipped:
            logger.warning("以下关节超时未标定: %s", skipped)
        if enc_positions:
            enc_summary = "; ".join(
                "%s enc=%.4f" % (j, e) for j, e in enc_positions.items()
            )
            logger.info("标定结束各关节 enc: %s", enc_summary)
        if persist and offsets:
            hardware.persist_zero_offsets(offsets)
        return offsets

    def _run_calibration_step(
        self,
        step: "CalibrationStep",
        step_idx: int,
        total_steps: int,
        hardware: CalibrationHardware,
        offsets: Dict[str, float],
        skipped: List[str],
        *,
        enc_positions: Optional[Dict[str, float]] = None,
        skip_on_timeout: bool,
    ) -> None:
        """执行单个关节的「到位 → 搜索 → 判停 → 回退」子流程。

        任一阶段（``move_to_pose`` / 搜索超时 / 卡死早停）抛出的 :class:`TimeoutError`
        会冒泡到 :meth:`calibrate`，由其决定「中止」还是「跳过继续下一个」。
        """
        detector = self.config.detector
        jn = step.target_joint
        hardware.move_to_pose(step.hold_pose, speed_scale=self.config.pose_speed_scale)
        hardware.sleep(self.config.settle_seconds)
        mode = self.config.search_mode
        if mode == "torque_damping":
            hardware.start_torque_damping_search(
                jn,
                search_direction_sign_for_step(step),
                float(self.config.torque_search_nm),
                float(self.config.torque_damping_nm_s),
            )
        elif mode == "velocity":
            hardware.start_velocity_search(jn, step.search_velocity)
        else:
            raise ValueError(
                "search_mode 必须是 'torque_damping' 或 'velocity'，当前为 %r" % (mode,)
            )

        t0 = time.monotonic()
        history: List[JointSample] = []
        _last_diag = t0
        _DIAG_INTERVAL = 2.0
        search_start_pos: Optional[float] = None
        # 动态 effort 基线：在 effort_baseline_seconds 窗口内取 |effort| 最大值
        # （关节自由加速阶段的最高 effort），用于后续做「增量判停」。
        effort_baseline: Optional[float] = None
        _baseline_max = 0.0
        # 「卡滞早停」计时：除 near_limit 以外全部通过，但 dist2stop 持续超标时开始计时
        _stuck_since: Optional[float] = None
        while True:
            sample = hardware.read_sample(jn)
            if search_start_pos is None:
                search_start_pos = sample.encoder_position
            history.append(sample)
            history = [
                entry
                for entry in history
                if sample.timestamp - entry.timestamp <= detector.stall_time_seconds
            ]

            # 更新基线：只在前 effort_baseline_seconds 采样窗口内累积最大 |effort|
            elapsed_total = time.monotonic() - t0
            if (
                detector.effort_rise_nm > 0.0
                and elapsed_total <= detector.effort_baseline_seconds
            ):
                _baseline_max = max(_baseline_max, abs(sample.motor_current))
            elif effort_baseline is None and detector.effort_rise_nm > 0.0:
                effort_baseline = _baseline_max

            threshold = self.config.current_thresholds[jn]
            if self._stopped_on_hard_limit(
                history,
                threshold,
                search_start_pos,
                stop_angle=step.stop_angle,
                joint_name=jn,
                effort_baseline=effort_baseline,
            ):
                hardware.stop_joint(jn)
                hardware.sleep(detector.backoff_seconds)
                offset = self.compute_zero_offset(
                    jn,
                    encoder_position=sample.encoder_position,
                    reference_angle=step.stop_angle,
                )
                offsets[jn] = offset
                if enc_positions is not None:
                    enc_positions[jn] = sample.encoder_position
                hardware.apply_zero_offset(jn, offset)
                logger.info(
                    "[%d/%d] %s 检出硬限位: enc=%.4f → offset=%.6f rad",
                    step_idx, total_steps, jn, sample.encoder_position, offset,
                )
                release_pose = dict(step.hold_pose)
                release_pose[jn] = step.backoff_angle
                hardware.move_to_pose(release_pose, speed_scale=self.config.pose_speed_scale)
                hardware.sleep(self.config.settle_seconds)
                return

            elapsed = time.monotonic() - t0
            now_m = time.monotonic()

            # 「卡滞早停」：若除 near_limit 以外所有条件都已满足、离 stop_angle 仍超标，
            # 说明关节机械卡在中途（或搜索方向反了）。计时满 stuck_abort_seconds 立即失败。
            if detector.stuck_abort_seconds > 0.0 and detector.max_expected_offset > 0.0:
                stuck_now = self._stopped_on_hard_limit(
                    history,
                    threshold,
                    search_start_pos,
                    stop_angle=None,  # 关掉 near_limit 检查
                    joint_name=jn,
                    effort_baseline=effort_baseline,
                )
                if stuck_now:
                    if _stuck_since is None:
                        _stuck_since = now_m
                    elif now_m - _stuck_since >= detector.stuck_abort_seconds:
                        hardware.stop_joint(jn)
                        sample_now = history[-1]
                        sign = float(self.config.encoder_signs.get(jn, 1.0))
                        dist = abs(sign * sample_now.encoder_position - step.stop_angle)
                        logger.warning(
                            "[%d/%d] %s 卡滞早停 (%.1fs): enc=%.4f 距 stop=%.4f "
                            "差 %.4f rad > max_expected_offset=%.3f",
                            step_idx, total_steps, jn,
                            detector.stuck_abort_seconds,
                            sample_now.encoder_position, step.stop_angle,
                            dist, detector.max_expected_offset,
                        )
                        raise TimeoutError(
                            f"Hard-stop search stuck mid-travel for {jn} "
                            f"(enc={sample_now.encoder_position:.4f}, stop={step.stop_angle:.4f})"
                        )
                else:
                    _stuck_since = None

            if now_m - _last_diag >= _DIAG_INTERVAL:
                _last_diag = now_m
                self._log_search_diagnostics(
                    jn, history, threshold, step_idx, total_steps, elapsed,
                    search_start_pos=search_start_pos,
                    stop_angle=step.stop_angle,
                    effort_baseline=effort_baseline,
                )

            if elapsed > detector.sample_timeout_seconds:
                hardware.stop_joint(jn)
                # 超时前打印一次最终诊断，帮助现场定位失败原因
                self._log_search_diagnostics(
                    jn, history, threshold, step_idx, total_steps, elapsed,
                    search_start_pos=search_start_pos,
                    stop_angle=step.stop_angle,
                    effort_baseline=effort_baseline,
                )
                raise TimeoutError(f"Hard-stop search timed out for {jn}")


def plan_summary(plan: CalibrationPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        required=True,
        help="Path to URDF, for example casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf",
    )
    parser.add_argument("--arm", choices=("left", "right"), required=True, help="Arm side to plan for.")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the recommended hard-stop calibration plan as JSON.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    plan = build_default_arm_calibration_plan(args.urdf, args.arm)

    if args.print_plan:
        print(plan_summary(plan))
        return 0

    print(
        "This module contains the runtime skeleton only. "
        "Instantiate HardStopCalibrator with a project-specific hardware adapter to execute calibration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
