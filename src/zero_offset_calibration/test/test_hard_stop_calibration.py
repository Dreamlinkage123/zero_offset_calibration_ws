"""纯规划层单元测试：不依赖 rclpy / mujoco / numpy。

路径解析
--------
标定计划所需的 URDF 通过 :mod:`zero_offset_calibration._paths` 统一查找：
colcon test 时从 ``share/`` 解析；直接 ``pytest`` 源码时回落到
``src/zero_offset_calibration/casbot_band_urdf/urdf/...``。
"""

import tempfile
import unittest
from pathlib import Path

from zero_offset_calibration._paths import default_urdf_path
from zero_offset_calibration.hard_stop_calibration import (
    REST_ROLL_RAD,
    HardStopCalibrator,
    HardStopCalibratorConfig,
    HardStopDetectorConfig,
    JointSample,
    arm_joint_names,
    arm_reset_waypoints,
    arm_rest_pose,
    arm_setup_waypoints,
    build_default_arm_calibration_plan,
    detect_instrument_from_urdf_path,
    detect_instrument_from_xml_path,
    normalize_plan_instrument,
    parse_joint_limits,
    search_direction_sign_for_step,
    write_zero_offsets_yaml,
)


URDF_PATH = default_urdf_path("CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf")


class CalibrationPlanTests(unittest.TestCase):
    def test_parse_joint_limits_contains_right_arm_joint(self) -> None:
        limits = parse_joint_limits(URDF_PATH)
        self.assertIn("right_shoulder_roll_joint", limits)
        # bass URDF 已按实测关节限位统一更新
        self.assertAlmostEqual(limits["right_shoulder_roll_joint"].lower, -3.14159265, places=4)
        self.assertAlmostEqual(limits["right_shoulder_roll_joint"].upper, 0.34906585, places=4)

    def test_left_and_right_plans_have_seven_steps(self) -> None:
        left_plan = build_default_arm_calibration_plan(URDF_PATH, "left")
        right_plan = build_default_arm_calibration_plan(URDF_PATH, "right")
        self.assertEqual(len(left_plan.steps), 7)
        self.assertEqual(len(right_plan.steps), 7)

    def test_side_specific_roll_stop_selection(self) -> None:
        left_plan = build_default_arm_calibration_plan(URDF_PATH, "left")
        right_plan = build_default_arm_calibration_plan(URDF_PATH, "right")

        left_roll = next(step for step in left_plan.steps if step.target_joint == "left_shoulder_roll_joint")
        right_roll = next(step for step in right_plan.steps if step.target_joint == "right_shoulder_roll_joint")

        self.assertEqual(left_roll.stop_side, "upper")
        self.assertAlmostEqual(left_roll.stop_angle, 3.14159265, places=4)
        self.assertEqual(right_roll.stop_side, "lower")
        self.assertAlmostEqual(right_roll.stop_angle, -3.14159265, places=4)

    def test_search_direction_sign_matches_search_velocity(self) -> None:
        plan = build_default_arm_calibration_plan(URDF_PATH, "right")
        for step in plan.steps:
            v = step.search_velocity
            s = search_direction_sign_for_step(step)
            self.assertEqual(s, 1 if v > 0 else (-1 if v < 0 else 1))

    def test_zero_offset_formula_respects_encoder_sign(self) -> None:
        calibrator = HardStopCalibrator(
            HardStopCalibratorConfig(
                encoder_signs={"joint_a": -1.0},
                current_thresholds={"joint_a": 5.0},
            )
        )
        offset = calibrator.compute_zero_offset("joint_a", encoder_position=1.2, reference_angle=-0.4)
        self.assertAlmostEqual(offset, 0.8, places=6)

    def test_write_zero_offsets_yaml_roundtrip_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.yaml"
            data = {"right_wrist_pitch_joint": -0.0123, "left_elbow_pitch_joint": 0.0}
            write_zero_offsets_yaml(p, data, header_lines=())
            text = p.read_text(encoding="utf-8")
            self.assertIn("left_elbow_pitch_joint: 0.0", text)
            self.assertIn("right_wrist_pitch_joint:", text)
            self.assertTrue(p.suffix in {".yaml", ".yml"} or p.name.endswith("yaml"))


class _FakeHardware:
    """最小硬件适配器：指定某些关节在 move_to_pose 或 search 时抛 TimeoutError，
    其它关节立即满足判停条件，用于验证 ``skip_on_timeout`` 覆盖各类超时路径。"""

    def __init__(
        self,
        *,
        fail_move_joints: set,
        fail_search_joints: set,
        hit_position: float = 0.0,
    ) -> None:
        self.fail_move_joints = set(fail_move_joints)
        self.fail_search_joints = set(fail_search_joints)
        self.hit_position = float(hit_position)
        self._t = 0.0
        self.applied_offsets: dict = {}
        self.persisted: dict = {}
        self.stops: list = []

    def move_to_pose(self, pose, speed_scale):  # noqa: ARG002
        # 只在 pose 要求「把故障关节从 home 推到非零位置」时失败，
        # 这样非故障关节步骤的 hold_pose / release_pose 仍可通过（其故障关节目标=0）
        for j in self.fail_move_joints:
            if abs(pose.get(j, 0.0)) > 1e-3:
                raise TimeoutError(f"move_to_pose stuck: {j}")

    def sleep(self, seconds: float) -> None:
        self._t += max(0.0, seconds)

    def start_torque_damping_search(self, *_a, **_k) -> None:
        self._t = 0.0

    def start_velocity_search(self, *_a, **_k) -> None:
        self._t = 0.0

    def stop_joint(self, joint_name: str) -> None:
        self.stops.append(joint_name)

    def read_sample(self, joint_name: str) -> JointSample:
        self._t += 0.02
        if joint_name in self.fail_search_joints:
            # 永远不满足判停：速度大于 velocity_epsilon，且 position 样本也在变化
            return JointSample(
                encoder_position=self.hit_position + self._t,  # 位置持续变化 → position_window_epsilon 触发 False
                estimated_velocity=1000.0,
                motor_current=10.0,
                timestamp=self._t,
            )
        # 模拟已到达 hit_position 且完全静止
        return JointSample(
            encoder_position=self.hit_position,
            estimated_velocity=0.0,
            motor_current=10.0,
            timestamp=self._t,
        )

    def apply_zero_offset(self, joint_name: str, offset: float) -> None:
        self.applied_offsets[joint_name] = offset

    def persist_zero_offsets(self, offsets) -> None:
        self.persisted = dict(offsets)


class CalibrateTimeoutHandlingTests(unittest.TestCase):
    """skip_on_timeout=True 时，任一关节的任何超时都不应中断整臂标定。"""

    def _build_calibrator(self, plan) -> HardStopCalibrator:
        signs = {step.target_joint: 1.0 for step in plan.steps}
        thresholds = {step.target_joint: 5.0 for step in plan.steps}
        cfg = HardStopCalibratorConfig(
            encoder_signs=signs,
            current_thresholds=thresholds,
            # 判停门槛调到最宽松，让 _FakeHardware 的静止样本能立刻命中
            detector=HardStopDetectorConfig(
                velocity_epsilon=10.0,
                position_window_epsilon=10.0,
                # 采样步进 ≈0.02s / 次，需 ≥6 次迭代让 stall 窗口 (0.1s) 内积累足够样本
                stall_time_seconds=0.1,
                sample_timeout_seconds=0.5,
                backoff_seconds=0.0,
                min_search_travel=0.0,
                max_expected_offset=0.0,  # 关闭几何检查，允许 hit_position 任意
                effort_rise_nm=0.0,
                stuck_abort_seconds=0.0,
            ),
            settle_seconds=0.0,
        )
        return HardStopCalibrator(cfg)

    def test_move_to_pose_timeout_does_not_abort_loop(self) -> None:
        """核心契约：某关节 move_to_pose 卡死时 calibrate() 不应整体抛出，
        而是把该关节记为 skipped，循环继续到后续步骤。"""
        plan = build_default_arm_calibration_plan(URDF_PATH, "right")
        joints = [step.target_joint for step in plan.steps]
        # 让最后一个关节（wrist_roll，通常只在本步 pose 中为非零）卡死，
        # 使得其它步骤的 hold_pose 不会触发该关节的故障
        stuck_joint = joints[-1]
        hw = _FakeHardware(fail_move_joints={stuck_joint}, fail_search_joints=set())
        calibrator = self._build_calibrator(plan)

        offsets = calibrator.calibrate(plan, hw, persist=False, skip_on_timeout=True)

        self.assertNotIn(stuck_joint, offsets)
        # 至少应完成一部分非故障关节（证明循环没有被提前中断）
        self.assertGreaterEqual(len(offsets), 1)

    def test_search_timeout_skips_and_continues(self) -> None:
        plan = build_default_arm_calibration_plan(URDF_PATH, "right")
        joints = [step.target_joint for step in plan.steps]
        stuck_joint = joints[3]
        hw = _FakeHardware(fail_move_joints=set(), fail_search_joints={stuck_joint})
        calibrator = self._build_calibrator(plan)

        offsets = calibrator.calibrate(plan, hw, persist=False, skip_on_timeout=True)

        self.assertNotIn(stuck_joint, offsets)
        for j in joints:
            if j != stuck_joint:
                self.assertIn(j, offsets)

    def test_near_limit_alone_confirms_when_effort_rise_fails(self) -> None:
        """torque-damping 搜索到位后 effort 常常低于 baseline+rise；
        只要几何近邻（near_limit）已确认，就应该判停通过（OR 逻辑）。"""
        from zero_offset_calibration.hard_stop_calibration import (
            HardStopCalibrator,
            HardStopCalibratorConfig,
            HardStopDetectorConfig,
            JointSample,
        )

        cfg = HardStopCalibratorConfig(
            encoder_signs={"j": 1.0},
            current_thresholds={"j": 0.0},  # 关掉绝对电流门 → 启用 effort_rise
            detector=HardStopDetectorConfig(
                velocity_epsilon=0.1,
                position_window_epsilon=0.02,
                stall_time_seconds=0.1,
                sample_timeout_seconds=5.0,
                min_search_travel=0.0,
                max_expected_offset=0.60,
                effort_baseline_seconds=0.1,
                effort_rise_nm=0.15,
                stuck_abort_seconds=0.0,
            ),
        )
        calib = HardStopCalibrator(cfg)
        # 构造 stall 样本：位置 = stop_angle（near_limit OK），但 |eff|=0.1 < baseline(0.78)+rise → eff_rise 失败
        history = [
            JointSample(encoder_position=1.567, estimated_velocity=0.0,
                        motor_current=0.1, timestamp=t)
            for t in (0.0, 0.05, 0.10, 0.15)
        ]
        ok = calib._stopped_on_hard_limit(
            history,
            current_threshold=0.0,
            search_start_pos=1.30,
            stop_angle=1.567,
            joint_name="j",
            effort_baseline=0.78,
        )
        self.assertTrue(ok, "near_limit 已 OK 时应当接受，不应被 effort_rise 反证一票否决")

    def test_effort_rise_alone_confirms_when_near_limit_out_of_range(self) -> None:
        """反向验证：如果 near_limit 超出阈值但 effort_rise 满足，也应当通过。"""
        from zero_offset_calibration.hard_stop_calibration import (
            HardStopCalibrator,
            HardStopCalibratorConfig,
            HardStopDetectorConfig,
            JointSample,
        )

        cfg = HardStopCalibratorConfig(
            encoder_signs={"j": 1.0},
            current_thresholds={"j": 0.0},
            detector=HardStopDetectorConfig(
                velocity_epsilon=0.1,
                position_window_epsilon=0.02,
                stall_time_seconds=0.1,
                sample_timeout_seconds=5.0,
                min_search_travel=0.0,
                max_expected_offset=0.3,  # near_limit 会判 NO（距 stop 1.0 rad）
                effort_baseline_seconds=0.1,
                effort_rise_nm=0.15,
                stuck_abort_seconds=0.0,
            ),
        )
        calib = HardStopCalibrator(cfg)
        history = [
            JointSample(encoder_position=0.5, estimated_velocity=0.0,
                        motor_current=1.0, timestamp=t)  # |eff|=1.0 > 0.3+0.15
            for t in (0.0, 0.05, 0.10, 0.15)
        ]
        ok = calib._stopped_on_hard_limit(
            history,
            current_threshold=0.0,
            search_start_pos=0.3,
            stop_angle=1.5,
            joint_name="j",
            effort_baseline=0.3,
        )
        self.assertTrue(ok, "effort_rise 已 OK 时应当接受，即使 near_limit 距限位超标")

    def test_timeout_without_skip_flag_still_raises(self) -> None:
        plan = build_default_arm_calibration_plan(URDF_PATH, "right")
        stuck_joint = plan.steps[0].target_joint
        hw = _FakeHardware(fail_move_joints={stuck_joint}, fail_search_joints=set())
        calibrator = self._build_calibrator(plan)

        with self.assertRaises(TimeoutError):
            calibrator.calibrate(plan, hw, persist=False, skip_on_timeout=False)


class InstrumentDetectionTests(unittest.TestCase):
    """与 :mod:`mujoco_hard_stop_calibration` 的乐器映射保持一致。"""

    def test_detect_instrument_from_urdf_path(self) -> None:
        cases = [
            ("CASBOT02_ENCOS_7dof_shell_20251015_P1L.urdf", ""),
            ("CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf", "bass"),
            ("CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf", "guitar"),
            ("CASBOT02_ENCOS_7dof_shell_20251015_P1L_keyboard.urdf", "keyboard"),
            ("/abs/path/to/whatever_BASS.urdf", "bass"),
            ("no_suffix.urdf", ""),
        ]
        for filename, expected in cases:
            with self.subTest(filename=filename):
                self.assertEqual(detect_instrument_from_urdf_path(filename), expected)

    def test_detect_instrument_from_xml_path_matches_urdf_rules(self) -> None:
        """MuJoCo 仿真与真机使用同一套 stem 后缀规则，仅扩展名不同。"""
        cases = [
            ("CASBOT02_ENCOS_7dof_shell_20251015_P1L.xml", ""),
            ("xml/CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.xml", "bass"),
            (".../P1L_guitar.xml", "guitar"),
            ("/x/y/z/P1L_keyboard.xml", "keyboard"),
        ]
        for filename, expected in cases:
            with self.subTest(filename=filename):
                self.assertEqual(detect_instrument_from_xml_path(filename), expected)
        self.assertEqual(
            detect_instrument_from_urdf_path("a_bass.urdf"),
            detect_instrument_from_xml_path("a_bass.xml"),
        )

    def test_normalize_plan_instrument_aliases(self) -> None:
        # 与 mujoco 适配器的 _PLAN_INSTRUMENT_MAP 一致：keyboard 走 guitar 高抬臂轨迹
        self.assertEqual(normalize_plan_instrument("bass"), "bass")
        self.assertEqual(normalize_plan_instrument("guitar"), "guitar")
        self.assertEqual(normalize_plan_instrument("keyboard"), "guitar")
        self.assertEqual(normalize_plan_instrument(""), "")
        self.assertEqual(normalize_plan_instrument("none"), "")
        self.assertEqual(normalize_plan_instrument("UNKNOWN"), "")
        self.assertEqual(normalize_plan_instrument("BASS"), "bass")


class ArmRestAndWaypointTests(unittest.TestCase):
    """共享的 rest pose / setup waypoint / reset waypoint 行为；
    真机适配器 (:mod:`ros2_upper_body_hardware`) 与仿真适配器
    (:mod:`mujoco_hard_stop_calibration`) 都依赖这些函数生成轨迹，必须一致。"""

    def test_arm_rest_pose_outward_roll(self) -> None:
        left = arm_rest_pose("left")
        right = arm_rest_pose("right")
        self.assertEqual(set(left.keys()), set(arm_joint_names("left")))
        self.assertEqual(set(right.keys()), set(arm_joint_names("right")))
        # 仅 shoulder_roll 偏置外展 10°，其它关节为 0
        self.assertAlmostEqual(left["left_shoulder_roll_joint"], REST_ROLL_RAD)
        self.assertAlmostEqual(right["right_shoulder_roll_joint"], -REST_ROLL_RAD)
        for j, v in left.items():
            if j != "left_shoulder_roll_joint":
                self.assertEqual(v, 0.0)
        for j, v in right.items():
            if j != "right_shoulder_roll_joint":
                self.assertEqual(v, 0.0)

    def test_setup_waypoints_no_instrument_is_direct(self) -> None:
        plan = build_default_arm_calibration_plan(URDF_PATH, "right", instrument="")
        wps = arm_setup_waypoints("right", plan.neutral_pose, instrument="")
        # 裸机 neutral 的 |pitch|<1.5，不需要中间 waypoint
        self.assertEqual(len(wps), 1)
        for j in arm_joint_names("right"):
            self.assertAlmostEqual(wps[0][j], plan.neutral_pose.get(j, 0.0))

    def test_setup_waypoints_with_bass_uses_three_step_outward_then_lift(self) -> None:
        plan = build_default_arm_calibration_plan(URDF_PATH, "right", instrument="bass")
        wps = arm_setup_waypoints("right", plan.neutral_pose, instrument="bass")
        self.assertEqual(len(wps), 3, "高抬臂 (|pitch|>1.5) 必须有 WP1/WP2/neutral 三步")
        wp1, wp2, final = wps

        target_pitch = plan.neutral_pose["right_shoulder_pitch_joint"]
        target_roll = plan.neutral_pose["right_shoulder_roll_joint"]
        target_elbow = plan.neutral_pose["right_elbow_pitch_joint"]
        # WP1: 仅展 roll，pitch/elbow 维持 0（避免前臂抬起时扫过琴身）
        self.assertEqual(wp1["right_shoulder_pitch_joint"], 0.0)
        self.assertEqual(wp1["right_elbow_pitch_joint"], 0.0)
        self.assertAlmostEqual(wp1["right_shoulder_roll_joint"], target_roll)
        # WP2: 抬 pitch，elbow 仍为 0（手臂已外展，可绕过乐器）
        self.assertAlmostEqual(wp2["right_shoulder_pitch_joint"], target_pitch)
        self.assertEqual(wp2["right_elbow_pitch_joint"], 0.0)
        self.assertAlmostEqual(wp2["right_shoulder_roll_joint"], target_roll)
        # 终点：完整 neutral（含 elbow 弯曲）
        self.assertAlmostEqual(final["right_shoulder_pitch_joint"], target_pitch)
        self.assertAlmostEqual(final["right_elbow_pitch_joint"], target_elbow)

    def test_reset_waypoints_no_instrument_neutral_then_rest(self) -> None:
        plan = build_default_arm_calibration_plan(URDF_PATH, "left", instrument="")
        wps = arm_reset_waypoints("left", plan.neutral_pose, instrument="")
        self.assertEqual(len(wps), 2)
        # 终点为 rest（外展 10°）
        self.assertAlmostEqual(wps[-1]["left_shoulder_roll_joint"], REST_ROLL_RAD)
        for j in arm_joint_names("left"):
            if j != "left_shoulder_roll_joint":
                self.assertEqual(wps[-1][j], 0.0)

    def test_reset_waypoints_with_guitar_three_step_inverse(self) -> None:
        plan = build_default_arm_calibration_plan(URDF_PATH, "left", instrument="guitar")
        wps = arm_reset_waypoints("left", plan.neutral_pose, instrument="guitar")
        self.assertEqual(len(wps), 3, "高抬臂收尾必须 neutral → rev-WP2 → rest")
        target_pitch = plan.neutral_pose["left_shoulder_pitch_joint"]
        target_roll = plan.neutral_pose["left_shoulder_roll_joint"]
        # 第 1 步：neutral
        self.assertAlmostEqual(wps[0]["left_shoulder_pitch_joint"], target_pitch)
        # 第 2 步：rev-WP2 — pitch 降到 0，roll 仍维持外展到 neutral 值
        self.assertEqual(wps[1]["left_shoulder_pitch_joint"], 0.0)
        self.assertAlmostEqual(wps[1]["left_shoulder_roll_joint"], target_roll)
        self.assertEqual(wps[1]["left_elbow_pitch_joint"], 0.0)
        # 第 3 步：rest — roll 收回到 +REST_ROLL_RAD
        self.assertAlmostEqual(wps[2]["left_shoulder_roll_joint"], REST_ROLL_RAD)
        self.assertEqual(wps[2]["left_shoulder_pitch_joint"], 0.0)

    def test_setup_and_reset_waypoints_only_contain_target_arm_joints(self) -> None:
        """waypoint 仅包含本臂 7 个关节，便于真机适配器只下发必要关节。"""
        plan = build_default_arm_calibration_plan(URDF_PATH, "right", instrument="bass")
        for wp in arm_setup_waypoints("right", plan.neutral_pose, instrument="bass"):
            self.assertEqual(set(wp.keys()), set(arm_joint_names("right")))
        for wp in arm_reset_waypoints("right", plan.neutral_pose, instrument="bass"):
            self.assertEqual(set(wp.keys()), set(arm_joint_names("right")))


if __name__ == "__main__":
    unittest.main()
