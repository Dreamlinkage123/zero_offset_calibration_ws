"""CASBOT 上半身硬限位零偏校准包。

Modules
-------
- :mod:`zero_offset_calibration.hard_stop_calibration` — URDF 解析、标定计划、运行时骨架；无 ROS / MuJoCo 依赖。
- :mod:`zero_offset_calibration.mujoco_hard_stop_calibration` — MuJoCo 仿真适配器（需 ``mujoco``、``numpy``）。
- :mod:`zero_offset_calibration.ros2_upper_body_hardware` — ROS 2 真机适配器（需 ``rclpy``、``crb_ros_msg``）。
"""

from .hard_stop_calibration import (  # noqa: F401 re-export for convenience
    CalibrationHardware,
    CalibrationPlan,
    CalibrationStep,
    HardStopCalibrator,
    HardStopCalibratorConfig,
    HardStopDetectorConfig,
    JointLimit,
    JointSample,
    LEFT_ARM_JOINTS,
    REST_ROLL_RAD,
    RIGHT_ARM_JOINTS,
    arm_joint_names,
    arm_reset_waypoints,
    arm_rest_pose,
    arm_setup_waypoints,
    build_default_arm_calibration_plan,
    detect_instrument_from_urdf_path,
    detect_instrument_from_xml_path,
    normalize_plan_instrument,
    parse_joint_limits,
    plan_summary,
    preferred_stop_side,
    search_direction_sign_for_step,
    write_zero_offsets_yaml,
)

__all__ = [
    "CalibrationHardware",
    "CalibrationPlan",
    "CalibrationStep",
    "HardStopCalibrator",
    "HardStopCalibratorConfig",
    "HardStopDetectorConfig",
    "JointLimit",
    "JointSample",
    "LEFT_ARM_JOINTS",
    "REST_ROLL_RAD",
    "RIGHT_ARM_JOINTS",
    "arm_joint_names",
    "arm_reset_waypoints",
    "arm_rest_pose",
    "arm_setup_waypoints",
    "build_default_arm_calibration_plan",
    "detect_instrument_from_urdf_path",
    "detect_instrument_from_xml_path",
    "normalize_plan_instrument",
    "parse_joint_limits",
    "plan_summary",
    "preferred_stop_side",
    "search_direction_sign_for_step",
    "write_zero_offsets_yaml",
]
