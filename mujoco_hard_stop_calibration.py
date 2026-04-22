#!/usr/bin/env python3
"""在 MuJoCo 中复现上身硬限位零偏标定流程（与 `hard_stop_calibration.HardStopCalibrator` 对接）。

使用仓库内 `casbot_band_urdf/xml/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.xml`。
该模型腿部关节无执行器，逐仿真步将腿关节位姿/速度锁回初值，避免全身垮塌。

`read_sample` 使用**仿真时间**作为 `JointSample.timestamp`（与 `m.opt.timestep` 累加），以便
`HardStopDetectorConfig.stall_time_seconds` 在快算仿真中仍表示“在仿真中停留的时长”。

依赖：mujoco（>=3）, numpy。加 `--visualize` 时通过 `mujoco.viewer.launch_passive` 在独立线程
打开交互仿真窗口，主线程在标定循环里定期 `sync()` 更新画面；关闭窗口会中断并抛错。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np

import mujoco
import mujoco.viewer

from hard_stop_calibration import (
    CalibrationHardware,
    HardStopCalibrator,
    HardStopCalibratorConfig,
    HardStopDetectorConfig,
    JointLimit,
    JointSample,
    arm_joint_names,
    build_default_arm_calibration_plan,
    parse_joint_limits,
    plan_summary,
    write_zero_offsets_yaml,
)


def _jnt_name(m: mujoco.MjModel, j: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""


def _act_name(m: mujoco.MjModel, a: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""


@dataclass
class MujocoCalibConfig:
    model_xml: Path
    urdf_path: Path
    arm: str
    # PD for <motor> actuators (N·m scale per model)
    kp: float = 150.0
    kd: float = 18.0
    max_motor_torque: float = 72.0
    move_max_steps: int = 12000
    move_tol: float = 0.08
    settle_steps: int = 80
    # 标定大姿态到位：每步对 _q_des 的限速 (rad/s)，与 HardStopCalibrator 的 pose_speed_scale 相乘
    pose_max_vel_rad_s: float = 0.18
    # 靠限位搜索的关节角速度 = plan.search_velocity * 此系数
    # 注意：过小会导致手臂在躯干/碰撞处久停，易误触发判停、零偏明显偏离真限位，仅建议可视化调试用低值
    search_velocity_scale: float = 1.0
    # 到位失败时避免一次性 qpos 硬塞：分多步从当前值插到目标
    pose_snap_blend_steps: int = 36
    pose_snap_damp_steps: int = 360
    # 模拟原始编码器读数偏差（rad），标定后应估计出约 -bias（sign=+1 时）
    encoder_bias_rad: Dict[str, float] = field(default_factory=dict)
    # 贴近关节 range 时把判停力矩抬到至少该值，避免腕部等小关节传感器力矩偏低导致判停失败
    near_limit_torque_floor: float = 1.2
    # `mujoco.viewer.launch_passive` 返回的 Handle；非 None 时每步同步画面
    viewer_handle: Optional[Any] = None
    # 每多少步物理仿真调用一次 viewer.sync()，避免过慢（数值越小画面越流畅）
    viewer_sync_every: int = 4


class MujocoArmCalibrationHardware:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, cfg: MujocoCalibConfig, joint_limits: Dict[str, JointLimit], arm_joints: tuple[str, ...]):
        self.m = m
        self.d = d
        self.cfg = cfg
        self._limits = {j: joint_limits[j] for j in arm_joints}
        self._arm_joints = arm_joints
        self._sim_time = 0.0
        self._q_des: Dict[str, float] = {}
        self._actuator_is_motor: Dict[str, bool] = {}
        self._leg_lock: List[tuple[int, int, float]] = []  # (qpos_adr, dof_adr, q0)
        self._tau_adr: Dict[str, int] = {}
        self._qpos_adr: Dict[str, int] = {}
        self._dof_adr: Dict[str, int] = {}
        self._search_joint: Optional[str] = None
        self._search_vel: float = 0.0
        self._search_active = False
        # "none" | "velocity" | "torque"
        self._search_mode: str = "none"
        self._td_sign: float = 1.0
        self._td_tau: float = 0.0
        self._td_damp: float = 0.0
        self._jnt_range: Dict[str, tuple[float, float]] = {}
        self._viewer = cfg.viewer_handle
        self._viewer_physics_step = 0
        for j in range(m.njnt):
            jn = _jnt_name(m, j)
            if not jn or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            qad = m.jnt_qposadr[j]
            dof = m.jnt_dofadr[j]
            self._qpos_adr[jn] = int(qad)
            self._dof_adr[jn] = int(dof)
            self._jnt_range[jn] = (float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1]))

        for a in range(m.nu):
            an = _act_name(m, a)
            # motor: gear 1, position: kp 30
            self._actuator_is_motor[an] = bool(m.actuator_gainprm[a, 0] < 5.0)

        for i in range(m.nsensor):
            sn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
            if sn.endswith("_tau") and m.sensor_dim[i] == 1:
                jn = sn[: -len("_tau")]
                self._tau_adr[jn] = int(m.sensor_adr[i])

        mujoco.mj_resetData(m, d)
        self._setup_leg_lock()
        self._capture_initial_q_des()
        self._stabilize_idle()

    def _setup_leg_lock(self) -> None:
        m, d = self.m, self.d
        self._leg_lock = []
        for j in range(m.njnt):
            jn = _jnt_name(m, j)
            if not jn.startswith(("left_leg_", "right_leg_")):
                continue
            qad = m.jnt_qposadr[j]
            dof = m.jnt_dofadr[j]
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                w = 4
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                w = 7
            else:
                w = 1
            for k in range(w):
                self._leg_lock.append((qad + k, dof + k, float(d.qpos[qad + k])))
        mujoco.mj_forward(m, d)

    def _read_joint_q(self, name: str) -> float:
        adr = self._qpos_adr[name]
        return float(self.d.qpos[adr])

    def _capture_initial_q_des(self) -> None:
        for a in range(self.m.nu):
            an = _act_name(self.m, a)
            if an not in self._qpos_adr:
                continue
            self._q_des[an] = self._read_joint_q(an)
        for jn in self._arm_joints:
            if jn in self._q_des:
                continue
            if jn in self._qpos_adr:
                self._q_des[jn] = self._read_joint_q(jn)

    def _lock_legs(self) -> None:
        d = self.d
        for qad, dof, q0 in self._leg_lock:
            d.qpos[qad] = q0
            d.qvel[dof] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def _apply_ctrl(self) -> None:
        m, d = self.m, self.d
        for a in range(m.nu):
            an = _act_name(m, a)
            if an not in self._q_des:
                continue
            if an not in self._qpos_adr:
                continue
            qad = self._qpos_adr[an]
            q = d.qpos[qad]
            dof = self._dof_adr[an]
            qd = d.qvel[dof]
            tr = self.cfg.max_motor_torque
            if self._search_active and self._search_mode == "torque" and an == self._search_joint:
                tau = self._td_sign * self._td_tau - self._td_damp * qd
                d.ctrl[a] = float(np.clip(tau, -tr, tr))
                continue
            qgoal = self._q_des[an]
            if self._actuator_is_motor.get(an, True):
                e = qgoal - q
                tau = self.cfg.kp * e - self.cfg.kd * qd
                d.ctrl[a] = float(np.clip(tau, -tr, tr))
            else:
                d.ctrl[a] = float(qgoal)

    def _step_physics(self) -> None:
        self._apply_ctrl()
        mujoco.mj_step(self.m, self.d)
        self._lock_legs()
        self._sim_time += float(self.m.opt.timestep)
        vh = self._viewer
        if vh is not None:
            if not vh.is_running():
                raise RuntimeError("已关闭 MuJoCo 仿真窗口，标定中断")
            self._viewer_physics_step += 1
            if self._viewer_physics_step % max(1, int(self.cfg.viewer_sync_every)) == 0:
                vh.sync()

    def _stabilize_idle(self) -> None:
        for _ in range(self.cfg.settle_steps):
            self._step_physics()

    def _converged_pose(self, pose: Dict[str, float], tol: float) -> bool:
        for jn, qg in pose.items():
            if abs(self._read_joint_q(jn) - qg) > tol:
                return False
        return True

    def move_to_pose(self, pose: Dict[str, float], speed_scale: float) -> None:
        s = max(0.02, min(1.0, float(speed_scale)))
        dt = float(self.m.opt.timestep)
        v_max = self.cfg.pose_max_vel_rad_s * s
        q_goal = {k: float(v) for k, v in pose.items()}

        for _ in range(self.cfg.move_max_steps):
            for jn, qg in q_goal.items():
                if jn not in self._qpos_adr:
                    continue
                e = qg - self._q_des[jn]
                self._q_des[jn] += float(np.clip(e, -v_max * dt, v_max * dt))
            self._step_physics()
            if self._converged_pose(q_goal, self.cfg.move_tol):
                return
        # 仿真用：纯 PD 偶发无法到位，分多步在关节空间插值到命令角再阻尼，避免一步硬塞与躯干深穿
        d = self.d
        nblend = max(4, int(self.cfg.pose_snap_blend_steps))
        q0 = {jn: float(d.qpos[self._qpos_adr[jn]]) for jn in q_goal if jn in self._qpos_adr}
        for i in range(1, nblend + 1):
            t = i / float(nblend)
            for jn, qg in q_goal.items():
                if jn not in self._qpos_adr:
                    continue
                q1 = (1.0 - t) * q0.get(jn, self._q_des[jn]) + t * qg
                a = self._qpos_adr[jn]
                d.qpos[a] = q1
                self._q_des[jn] = q1
                if jn in self._dof_adr:
                    d.qvel[self._dof_adr[jn]] = 0.0
            mujoco.mj_forward(self.m, d)
            self._lock_legs()
        for _ in range(int(self.cfg.pose_snap_damp_steps)):
            self._step_physics()
        if not self._converged_pose(q_goal, self.cfg.move_tol * 2.0):
            raise TimeoutError("move_to_pose: 分步插值后仍未达到容差")

    def start_velocity_search(self, joint_name: str, velocity: float) -> None:
        self._search_mode = "velocity"
        self._search_active = True
        self._search_joint = joint_name
        self._search_vel = float(velocity) * self.cfg.search_velocity_scale

    def start_torque_damping_search(
        self, joint_name: str, sign: int, constant_torque: float, damping: float
    ) -> None:
        # τ = sign·T − b·q̇，由 _apply_ctrl 在搜索关节上覆盖 PD
        self._search_mode = "torque"
        self._search_active = True
        self._search_joint = joint_name
        self._td_sign = 1.0 if int(sign) >= 0 else -1.0
        self._td_tau = max(0.0, float(constant_torque))
        self._td_damp = max(0.0, float(damping))
        if joint_name in self._q_des:
            self._q_des[joint_name] = self._read_joint_q(joint_name)

    def read_sample(self, joint_name: str) -> JointSample:
        m, d = self.m, self.d
        if (
            self._search_active
            and self._search_mode == "velocity"
            and joint_name == self._search_joint
        ):
            jn = self._search_joint
            if jn is not None:
                lim = self._limits[jn]
                dt = float(m.opt.timestep)
                nxt = self._q_des[jn] + self._search_vel * dt
                nxt = float(np.clip(nxt, lim.lower, lim.upper))
                self._q_des[jn] = nxt
        self._step_physics()
        qad = self._qpos_adr[joint_name]
        dof = self._dof_adr[joint_name]
        q = float(d.qpos[qad])
        qd = float(d.qvel[dof])
        bias = self.cfg.encoder_bias_rad.get(joint_name, 0.0)
        enc = q + bias
        if joint_name in self._tau_adr:
            t = float(d.sensordata[self._tau_adr[joint_name]])
        else:
            t = 0.0
        t = abs(t)
        lo, hi = self._jnt_range.get(joint_name, (-1e9, 1e9))
        if min(abs(q - lo), abs(q - hi)) < 0.035 and abs(qd) < 0.35:
            t = max(t, self.cfg.near_limit_torque_floor)
        return JointSample(
            encoder_position=enc,
            estimated_velocity=qd,
            motor_current=t,
            timestamp=self._sim_time,
        )

    def stop_joint(self, joint_name: str) -> None:
        self._search_active = False
        self._search_mode = "none"
        if joint_name in self._qpos_adr:
            self._q_des[joint_name] = self._read_joint_q(joint_name)

    def apply_zero_offset(self, joint_name: str, offset: float) -> None:
        pass

    def persist_zero_offsets(self, offsets: Dict[str, float]) -> None:
        return

    def sleep(self, seconds: float) -> None:
        n = int(seconds / self.m.opt.timestep) + 1
        for _ in range(n):
            self._step_physics()


def _set_neutral_and_settle(
    plan,
    hardware: MujocoArmCalibrationHardware,
    arm: str,
    speed_scale: float,
) -> None:
    """先到中性/展臂等初始姿态，使用与标定相同的速度上限，避免开局瞬间甩臂穿模。"""
    pose: Dict[str, float] = dict(plan.neutral_pose)
    for j in arm_joint_names("left" if arm == "right" else "right"):
        if j in hardware._q_des:
            pose[j] = 0.0
    for name in ("waist_yaw_joint", "head_yaw_joint", "head_pitch_joint"):
        if name in hardware._q_des:
            pose[name] = 0.0
    hardware.move_to_pose(pose, speed_scale=speed_scale)
    for _ in range(max(hardware.cfg.settle_steps, 60)):
        hardware._step_physics()


def run_calibration(
    model_xml: Path,
    urdf_path: Path,
    arm: str,
    out_yaml: Path,
    encoder_bias: Optional[Dict[str, float]] = None,
    *,
    visualize: bool = False,
    viewer_sync_every: int = 4,
    pose_max_vel_rad_s: float = 0.14,
    search_velocity_scale: float = 1.0,
    settle_seconds: float = 0.2,
    pose_speed_scale: float = 0.35,
    search_mode: str = "torque_damping",
    torque_search_nm: float = 8.0,
    torque_damping_nm_s: float = 3.0,
) -> Dict[str, float]:
    m = mujoco.MjModel.from_xml_path(str(model_xml.resolve()))
    d = mujoco.MjData(m)

    def _execute_with_optional_viewer(viewer: Optional[Any]) -> Dict[str, float]:
        all_limits = parse_joint_limits(urdf_path)
        arm_list = arm_joint_names(arm)
        joint_limits = {n: all_limits[n] for n in arm_list}
        cfg = MujocoCalibConfig(
            model_xml=model_xml,
            urdf_path=urdf_path,
            arm=arm,
            encoder_bias_rad=encoder_bias or {},
            viewer_handle=viewer,
            viewer_sync_every=viewer_sync_every,
            pose_max_vel_rad_s=pose_max_vel_rad_s,
            search_velocity_scale=search_velocity_scale,
        )
        plan = build_default_arm_calibration_plan(urdf_path, arm)
        hw = MujocoArmCalibrationHardware(m, d, cfg, joint_limits, arm_list)
        _set_neutral_and_settle(plan, hw, arm, speed_scale=pose_speed_scale)
        det = HardStopDetectorConfig(
            stall_time_seconds=0.2,
            position_window_epsilon=0.015,
            velocity_epsilon=0.2,
            min_current_ratio=0.08,
            sample_timeout_seconds=60.0,
        )
        thresholds = {j: 0.35 for j in arm_list}
        signs = {j: 1.0 for j in arm_list}
        calib = HardStopCalibrator(
            HardStopCalibratorConfig(
                encoder_signs=signs,
                current_thresholds=thresholds,
                detector=det,
                pose_speed_scale=pose_speed_scale,
                settle_seconds=settle_seconds,
                search_mode=search_mode,
                torque_search_nm=torque_search_nm,
                torque_damping_nm_s=torque_damping_nm_s,
            )
        )
        mode = "交互可视化" if viewer is not None else "无头"
        print("MuJoCo 标定 [%s]: arm=%s, dt=%.4f" % (mode, arm, m.opt.timestep))
        if viewer is not None:
            try:
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_200,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "CASBOT02 零偏标定 (MuJoCo)",
                        "close window to cancel",
                    )
                )
            except (mujoco.UnexpectedError, AttributeError):
                pass
        hw.sleep(0.2)
        offsets = calib.calibrate(plan, cast(CalibrationHardware, hw), persist=False)
        write_zero_offsets_yaml(
            out_yaml,
            offsets,
            header_lines=(
                "MuJoCo hard-stop zero-offset calibration (radians).",
                f"model={model_xml.name} arm={arm}",
            ),
        )
        print("已写入: %s" % out_yaml)
        for k, v in sorted(offsets.items()):
            b = cfg.encoder_bias_rad.get(k, 0.0)
            if abs(b) > 1e-9:
                print("  %s  offset=%.6f  (注入 bias=%.6f, 预期 offset≈%.6f)" % (k, v, b, -b if signs[k] == 1.0 else b))
            else:
                print("  %s  offset=%.6f" % (k, v))
        return offsets

    if visualize:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            return _execute_with_optional_viewer(viewer)
    return _execute_with_optional_viewer(None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-xml",
        type=Path,
        default=Path("casbot_band_urdf/xml/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.xml"),
    )
    ap.add_argument(
        "--urdf",
        type=Path,
        default=Path("casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf"),
    )
    ap.add_argument("--arm", choices=("left", "right"), required=True)
    ap.add_argument("--out", type=Path, default=Path("zero_offsets_mujoco.yaml"))
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument(
        "--encoder-bias",
        type=Path,
        help="JSON: joint_name -> 模拟编码器常值偏差(rad)",
    )
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="用 mujoco.viewer.launch_passive 打开交互窗口，实时看标定过程",
    )
    ap.add_argument(
        "--viewer-sync-every",
        type=int,
        default=4,
        help="可视化时每多少步物理仿真 sync 一次（默认 4，越小越流畅越慢）",
    )
    ap.add_argument(
        "--pose-max-vel",
        type=float,
        default=0.14,
        help="move_to_pose 时关节角速度上限 (rad/s)，与 pose-speed-scale 相乘，主要减轻大幅摆臂穿模/过快",
    )
    ap.add_argument(
        "--search-vel-scale",
        type=float,
        default=1.0,
        help="靠限位搜索时对标定规划角速度的倍率。过小易在躯体碰撞面误触发判停，标定会不准；仅观察动画可设 0.4–0.7",
    )
    ap.add_argument(
        "--settle-seconds",
        type=float,
        default=0.2,
        help="每步标定中到位后等待时间（仿真秒）",
    )
    ap.add_argument(
        "--pose-speed-scale",
        type=float,
        default=0.35,
        help="传给标定核的 pose_speed_scale，与 --pose-max-vel 相乘，控制摆臂/到位快慢",
    )
    ap.add_argument(
        "--search-mode",
        choices=("torque_damping", "velocity"),
        default="torque_damping",
        help="找限位：torque_damping=恒力矩+阻尼(MuJoCo 为真力矩)；velocity=原角目标积分",
    )
    ap.add_argument(
        "--torque-nm",
        type=float,
        default=8.0,
        help="力矩模式：恒定力矩幅值 (N·m)，方向与规划 search_velocity 一致",
    )
    ap.add_argument(
        "--damping",
        type=float,
        default=3.0,
        help="力矩模式：阻尼系数 (N·m·s/rad)",
    )
    args = ap.parse_args()
    if args.print_plan:
        print(plan_summary(build_default_arm_calibration_plan(args.urdf, args.arm)))
        return 0
    enc_bias: Dict[str, float] = {}
    if args.encoder_bias and args.encoder_bias.is_file():
        enc_bias = {str(k): float(v) for k, v in json.loads(args.encoder_bias.read_text(encoding="utf-8")).items()}
    run_calibration(
        model_xml=args.model_xml,
        urdf_path=args.urdf,
        arm=args.arm,
        out_yaml=args.out,
        encoder_bias=enc_bias,
        visualize=bool(args.visualize),
        viewer_sync_every=max(1, int(args.viewer_sync_every)),
        pose_max_vel_rad_s=float(args.pose_max_vel),
        search_velocity_scale=float(args.search_vel_scale),
        settle_seconds=float(args.settle_seconds),
        pose_speed_scale=float(args.pose_speed_scale),
        search_mode=str(args.search_mode),
        torque_search_nm=float(args.torque_nm),
        torque_damping_nm_s=float(args.damping),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
