#!/usr/bin/env python3
"""Hard-stop based joint zero-offset calibration planner and runtime.

This module is intentionally split into two layers:
1. URDF-based planning: parse joint limits and generate recommended
   calibration steps for the left or right 7-DOF arm.
2. Runtime skeleton: execute those steps through a user-provided hardware
   adapter that knows how to talk to motors, encoders, and non-volatile
   storage on the real robot.

The repository only contains robot model assets, so the hardware-facing
methods are left abstract on purpose.
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
import re
from pathlib import Path
from typing import Dict, List, Mapping, Protocol


def write_zero_offsets_yaml(
    path: str | Path,
    offsets: Mapping[str, float],
    *,
    header_lines: tuple[str, ...] = (
        "Joint zero-offset calibration result (radians).",
        "joint_angle = encoder_sign * encoder_reading + offset",
    ),
) -> None:
    """Write offsets to a YAML file without requiring PyYAML.

    Joint names from this project are safe as YAML keys; others are quoted.
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


def rad_to_deg(value: float) -> float:
    return value * 180.0 / 3.141592653589793


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


@dataclass(frozen=True)
class JointLimit:
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
    encoder_position: float
    estimated_velocity: float
    motor_current: float
    timestamp: float


@dataclass(frozen=True)
class HardStopDetectorConfig:
    min_current_ratio: float = 0.30
    velocity_epsilon: float = 0.015
    position_window_epsilon: float = 0.003
    stall_time_seconds: float = 0.20
    sample_timeout_seconds: float = 10.0
    backoff_seconds: float = 0.20


@dataclass(frozen=True)
class HardStopCalibratorConfig:
    encoder_signs: Mapping[str, float]
    current_thresholds: Mapping[str, float]
    detector: HardStopDetectorConfig = field(default_factory=HardStopDetectorConfig)
    pose_speed_scale: float = 0.15
    settle_seconds: float = 0.30
    #: 找限位阶段：`"velocity"` 为关节角目标积分（原行为）；`"torque_damping"` 为恒力矩+阻尼
    search_mode: str = "torque_damping"
    # 力矩模式：τ = sign(搜索方向) * torque_search_nm - torque_damping_nm_s * 角速度（方向与规划中的 search_velocity 一致）
    torque_search_nm: float = 8.0
    torque_damping_nm_s: float = 3.0


class CalibrationHardware(Protocol):
    """Hardware-facing methods that must be provided by the integrator."""

    def move_to_pose(self, pose: Mapping[str, float], speed_scale: float) -> None:
        """Move all listed joints to the requested pose and block until done."""

    def read_sample(self, joint_name: str) -> JointSample:
        """Return a fresh encoder/current sample for the joint."""

    def start_velocity_search(self, joint_name: str, velocity: float) -> None:
        """Start a low-speed search motion toward the hard stop (position / velocity target mode)."""

    def start_torque_damping_search(
        self, joint_name: str, sign: int, constant_torque: float, damping: float
    ) -> None:
        """恒定力矩(幅值 constant_torque) + 与角速度成比例的阻尼，沿 sign 朝硬限位顶。

        `sign` 为 +1 或 -1，与 `CalibrationStep.search_velocity` 同号，表示在关节正方向/负方向上找限位。
        实机若仅有位置流，由适配器将 (τ, b) 等效为位置增量发布。
        """

    def stop_joint(self, joint_name: str) -> None:
        """Immediately stop the target joint."""

    def apply_zero_offset(self, joint_name: str, offset: float) -> None:
        """Apply the computed offset in RAM on the motion controller."""

    def persist_zero_offsets(self, offsets: Mapping[str, float]) -> None:
        """Write offsets to non-volatile storage."""

    def sleep(self, seconds: float) -> None:
        """Controller-aware sleep. Usually wraps time.sleep()."""


def parse_joint_limits(urdf_path: str | Path) -> Dict[str, JointLimit]:
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


def preferred_stop_side(arm: str, joint_name: str) -> int:
    """Return -1 for lower stop, +1 for upper stop."""

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


def default_neutral_pose(arm: str, joint_limits: Mapping[str, JointLimit]) -> Dict[str, float]:
    outward_roll = 0.60 if arm == "left" else -0.60
    pose = {
        f"{arm}_shoulder_pitch_joint": -1.10,
        f"{arm}_shoulder_roll_joint": outward_roll,
        f"{arm}_shoulder_yaw_joint": 0.0,
        f"{arm}_elbow_pitch_joint": -0.90,
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
) -> Dict[str, float]:
    pose: Dict[str, float] = dict(neutral_pose)

    shoulder_pitch = f"{arm}_shoulder_pitch_joint"
    shoulder_roll = f"{arm}_shoulder_roll_joint"
    shoulder_yaw = f"{arm}_shoulder_yaw_joint"
    elbow_pitch = f"{arm}_elbow_pitch_joint"
    wrist_yaw = f"{arm}_wrist_yaw_joint"
    wrist_pitch = f"{arm}_wrist_pitch_joint"
    wrist_roll = f"{arm}_wrist_roll_joint"

    outward_roll = 0.75 if arm == "left" else -0.75
    pose[shoulder_roll] = clamp(
        outward_roll,
        joint_limits[shoulder_roll].lower + 0.08,
        joint_limits[shoulder_roll].upper - 0.08,
    )
    pose[elbow_pitch] = clamp(
        -0.85,
        joint_limits[elbow_pitch].lower + 0.08,
        joint_limits[elbow_pitch].upper - 0.08,
    )

    if target_joint == shoulder_roll:
        pose[shoulder_pitch] = clamp(
            -1.20,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
    elif target_joint == shoulder_pitch:
        pose[shoulder_roll] = clamp(
            outward_roll,
            joint_limits[shoulder_roll].lower + 0.08,
            joint_limits[shoulder_roll].upper - 0.08,
        )
    elif target_joint == shoulder_yaw:
        pose[shoulder_pitch] = clamp(
            -1.00,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
    elif target_joint == elbow_pitch:
        pose[shoulder_pitch] = clamp(
            -1.00,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
        pose[shoulder_yaw] = 0.0
    elif target_joint in {wrist_yaw, wrist_pitch, wrist_roll}:
        pose[shoulder_pitch] = clamp(
            -0.95,
            joint_limits[shoulder_pitch].lower + 0.08,
            joint_limits[shoulder_pitch].upper - 0.08,
        )
        pose[shoulder_yaw] = 0.0
        pose[elbow_pitch] = clamp(
            -1.10,
            joint_limits[elbow_pitch].lower + 0.08,
            joint_limits[elbow_pitch].upper - 0.08,
        )

    pose[target_joint] = approach_angle
    return pose


def build_default_arm_calibration_plan(urdf_path: str | Path, arm: str) -> CalibrationPlan:
    all_limits = parse_joint_limits(urdf_path)
    joint_names = arm_joint_names(arm)
    joint_limits = {name: all_limits[name] for name in joint_names}
    neutral_pose = default_neutral_pose(arm, joint_limits)
    steps: List[CalibrationStep] = []

    for joint_name in joint_names:
        limit = joint_limits[joint_name]
        stop_direction = preferred_stop_side(arm, joint_name)
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
        hold_pose = step_hold_pose(arm, joint_name, approach_angle, neutral_pose, joint_limits)
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
    """与 `step.search_velocity` 同向，用于力矩/阻尼找限位。"""
    s = float(step.search_velocity)
    if s > 0.0:
        return 1
    if s < 0.0:
        return -1
    return 1


class HardStopCalibrator:
    def __init__(self, config: HardStopCalibratorConfig):
        self.config = config

    def compute_zero_offset(self, joint_name: str, encoder_position: float, reference_angle: float) -> float:
        sign = self.config.encoder_signs[joint_name]
        return reference_angle - sign * encoder_position

    def _stopped_on_hard_limit(
        self,
        history: List[JointSample],
        current_threshold: float,
    ) -> bool:
        detector = self.config.detector
        if len(history) < 2:
            return False

        newest = history[-1]
        oldest = history[0]
        duration = newest.timestamp - oldest.timestamp
        delta = abs(newest.encoder_position - oldest.encoder_position)
        current_ok = abs(newest.motor_current) >= current_threshold * detector.min_current_ratio
        velocity_ok = abs(newest.estimated_velocity) <= detector.velocity_epsilon
        position_ok = delta <= detector.position_window_epsilon
        # 仿真或离散采样时，时间跨度可能略小于标称 stall 窗口（例如步长 0.002s）
        stall_ok = duration >= detector.stall_time_seconds * 0.98
        return stall_ok and current_ok and velocity_ok and position_ok

    def calibrate(
        self,
        plan: CalibrationPlan,
        hardware: CalibrationHardware,
        persist: bool = False,
    ) -> Dict[str, float]:
        offsets: Dict[str, float] = {}
        detector = self.config.detector

        for step in plan.steps:
            hardware.move_to_pose(step.hold_pose, speed_scale=self.config.pose_speed_scale)
            hardware.sleep(self.config.settle_seconds)
            mode = self.config.search_mode
            if mode == "torque_damping":
                hardware.start_torque_damping_search(
                    step.target_joint,
                    search_direction_sign_for_step(step),
                    float(self.config.torque_search_nm),
                    float(self.config.torque_damping_nm_s),
                )
            elif mode == "velocity":
                hardware.start_velocity_search(step.target_joint, step.search_velocity)
            else:
                raise ValueError("search_mode 必须是 'torque_damping' 或 'velocity'，当前为 %r" % (mode,))

            t0 = time.monotonic()
            history: List[JointSample] = []
            while True:
                sample = hardware.read_sample(step.target_joint)
                history.append(sample)
                history = [
                    entry
                    for entry in history
                    if sample.timestamp - entry.timestamp <= detector.stall_time_seconds
                ]

                threshold = self.config.current_thresholds[step.target_joint]
                if self._stopped_on_hard_limit(history, threshold):
                    hardware.stop_joint(step.target_joint)
                    hardware.sleep(detector.backoff_seconds)
                    offset = self.compute_zero_offset(
                        step.target_joint,
                        encoder_position=sample.encoder_position,
                        reference_angle=step.stop_angle,
                    )
                    offsets[step.target_joint] = offset
                    hardware.apply_zero_offset(step.target_joint, offset)

                    release_pose = dict(step.hold_pose)
                    release_pose[step.target_joint] = step.backoff_angle
                    hardware.move_to_pose(release_pose, speed_scale=self.config.pose_speed_scale)
                    hardware.sleep(self.config.settle_seconds)
                    break

                if time.monotonic() - t0 > detector.sample_timeout_seconds:
                    hardware.stop_joint(step.target_joint)
                    raise TimeoutError(f"Hard-stop search timed out for {step.target_joint}")

        if persist:
            hardware.persist_zero_offsets(offsets)
        return offsets


def plan_summary(plan: CalibrationPlan) -> str:
    return json.dumps(plan.to_dict(), indent=2, sort_keys=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        required=True,
        help="Path to URDF, for example casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf",
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
