import tempfile
import unittest
from pathlib import Path

from hard_stop_calibration import (
    HardStopCalibrator,
    HardStopCalibratorConfig,
    build_default_arm_calibration_plan,
    parse_joint_limits,
    search_direction_sign_for_step,
    write_zero_offsets_yaml,
)


URDF_PATH = Path("casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf")


class CalibrationPlanTests(unittest.TestCase):
    def test_parse_joint_limits_contains_right_arm_joint(self) -> None:
        limits = parse_joint_limits(URDF_PATH)
        self.assertIn("right_shoulder_roll_joint", limits)
        self.assertAlmostEqual(limits["right_shoulder_roll_joint"].lower, -3.1416, places=4)
        self.assertAlmostEqual(limits["right_shoulder_roll_joint"].upper, 0.3491, places=4)

    def test_left_and_right_plans_have_seven_steps(self) -> None:
        left_plan = build_default_arm_calibration_plan(URDF_PATH, "left")
        right_plan = build_default_arm_calibration_plan(URDF_PATH, "right")
        self.assertEqual(len(left_plan.steps), 7)
        self.assertEqual(len(right_plan.steps), 7)

    def test_side_specific_roll_stop_selection(self) -> None:
        # shoulder_roll 选"贴身"窄端（±0.3491 rad 一侧），避免仿真中手臂经过
        # 躯干/头部碰撞体。详见 hard_stop_calibration.preferred_stop_side 注释。
        left_plan = build_default_arm_calibration_plan(URDF_PATH, "left")
        right_plan = build_default_arm_calibration_plan(URDF_PATH, "right")

        left_roll = next(step for step in left_plan.steps if step.target_joint == "left_shoulder_roll_joint")
        right_roll = next(step for step in right_plan.steps if step.target_joint == "right_shoulder_roll_joint")

        self.assertEqual(left_roll.stop_side, "lower")
        self.assertAlmostEqual(left_roll.stop_angle, -0.3491, places=4)
        self.assertEqual(right_roll.stop_side, "upper")
        self.assertAlmostEqual(right_roll.stop_angle, 0.3491, places=4)

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


if __name__ == "__main__":
    unittest.main()
