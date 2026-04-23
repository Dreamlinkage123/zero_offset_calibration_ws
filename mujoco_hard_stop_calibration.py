#!/usr/bin/env python3
"""在 MuJoCo 中复现上身硬限位零偏标定流程（与 `hard_stop_calibration.HardStopCalibrator` 对接）。

支持仓库内 `casbot_band_urdf/xml/` 下的多种模型（bare/guitar/bass），自动检测乐器
类型并生成对应的碰撞 primitive 和避障姿态。
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

import os
import numpy as np

# --record 的离屏渲染需要 GL 后端；无显示器时 GLFW 不可用，回退到 EGL
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

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


# 位掩码：让"手臂远端 ↔ 躯干/头部"发生物理碰撞而其它对保持静默。XML 默认
# 里 conaffinity=0 导致所有 body 互相穿透，运行时按此规则重写 contype/affinity：
#
# 位 1 (ARM_BIT=1): 标定侧 shoulder_roll_link 及以下（含 wrist/手指）
# 位 2 (BODY_BIT=2): waist_yaw_link / head_yaw_link / head_pitch_link
#
# 其它 body 的碰撞几何被显式清零，避免相邻骨段长期接触带来的数值噪声与卡死。
ARM_COLLISION_BIT = 1
BODY_COLLISION_BIT = 2

# 双臂自然下垂会与大腿干涉，rest pose 让 shoulder_roll 外展 10°（≈0.175 rad）
_REST_ROLL_RAD = 0.175

BODY_COLLISION_LINKS = (
    "waist_yaw_link",
    "head_yaw_link",
    "head_pitch_link",
)


_TORSO_COLLIDER = dict(
    name="waist_torso_collider",
    pos=(0.02, 0.0, 0.18),
    half_size=(0.11, 0.115, 0.19),
)

# 各乐器 mesh 后缀 → 额外碰撞 box 列表。通过对比 `waist_yaw_link.STL`（纯躯
# 干）与 `waist_yaw_link_{suffix}.STL` 的顶点差集在 Y/Z 切片下的分布估算。
# ─ 躯干 AABB（local）: X[-0.093,0.128] Y[±0.145] Z[-0.011,0.369]
#
# guitar:  琴身 Y[-0.40,+0.15] X[+0.09,+0.20] Z[-0.19,+0.07]
#          琴颈 Y[+0.15,+0.42] X≈+0.17 Z[+0.02,+0.14]
# bass:    琴身 Y[-0.32,+0.15] X[+0.07,+0.17] Z[-0.23,+0.05]
#          琴颈 Y[+0.15,+0.47] X[+0.18,+0.25] Z[+0.05,+0.23]
# 背带(guitar/bass 均有)不加碰撞——柔性织物，与左臂 hanging 空间完全重合。
_INSTRUMENT_COLLIDERS: dict[str, list[dict]] = {
    "guitar": [
        dict(name="guitar_body_collider",
             pos=(0.145, -0.125, -0.06), half_size=(0.055, 0.275, 0.13)),
        dict(name="guitar_neck_collider",
             pos=(0.162, 0.285, 0.08), half_size=(0.018, 0.135, 0.06)),
    ],
    "bass": [
        dict(name="bass_body_collider",
             pos=(0.12, -0.17, -0.09), half_size=(0.05, 0.15, 0.14)),
        dict(name="bass_neck_collider",
             pos=(0.22, 0.31, 0.15), half_size=(0.015, 0.16, 0.05)),
    ],
}


_CONTACT_INSTRUMENTS = frozenset({"guitar", "bass"})

# 与机器人 kinematic tree 无关、仅用于演奏/展示的独立 body（直属 worldbody），
# 标定时应删除以减少 DOF 和渲染负荷。
_STANDALONE_INSTRUMENT_BODIES = frozenset({"keyboard"})


def _detect_instrument(spec: "mujoco.MjSpec") -> str:
    """推断 XML 中的乐器类型：guitar / bass / keyboard / ""。

    1. waist_yaw_link mesh 名含 guitar/bass → 对应碰撞体需替换。
    2. worldbody 直接子 body 名为 keyboard → 独立乐器，标定时整体剥离。
    """
    for g in spec.geoms:
        parent = g.parent
        if parent is not None and parent.name == "waist_yaw_link":
            mesh_name = g.meshname if hasattr(g, "meshname") else ""
            if not mesh_name:
                try:
                    mesh_name = g.mesh.name if g.mesh is not None else ""
                except Exception:
                    mesh_name = ""
            for suffix in _INSTRUMENT_COLLIDERS:
                if suffix in mesh_name:
                    return suffix
    for b in spec.bodies:
        if b.name in _STANDALONE_INSTRUMENT_BODIES:
            return b.name
    return ""


def load_model_with_collision_filter(xml_path: Path, arm: str) -> "mujoco.MjModel":
    """用 MjSpec 在编译期重写 contype/conaffinity，打开"本侧手臂远端 ↔ 躯干/乐器
    代理体"的碰撞。

    设计要点：
    - 自动从 XML 中检测乐器类型（guitar / bass / keyboard / 空）。
    - 独立乐器 body（如 keyboard）整体剥离——它们的关节/geom 与标定无关，额外
      DOF 和 geom 会拖慢物理与渲染。
    - XML 中可能含有针对乐器接触的重型求解器设置（implicitfast, cone=elliptic,
      tolerance=1e-9 等），标定不需要，一律重置为 Euler + pyramidal 默认值。
    - waist_yaw_link 的原始 mesh collider 总是被删除——MuJoCo 对合并了乐器的 STL
      取凸包，远大于真实轮廓，会卡住手臂。删除后以 `_TORSO_COLLIDER` (box) 代理
      真实躯干，再由 `_INSTRUMENT_COLLIDERS[instrument]` 补上琴身/琴颈 box。
    - 背带(strap)区域不加碰撞——柔性织物，与左臂 hanging 空间完全重合。
    - MuJoCo 的 pair-filter 在 compile() 阶段缓存，运行时改 `m.geom_contype` 无效，
      因此必须在 MjSpec 上修改再编译。
    """
    spec = mujoco.MjSpec.from_file(str(xml_path.resolve()))
    instrument = _detect_instrument(spec)

    # --- 剥离独立乐器 body（如电子琴的琴键 body-tree） ---
    n_stripped = 0
    for b in list(spec.bodies):
        if b.name in _STANDALONE_INSTRUMENT_BODIES:
            spec.delete(b)
            n_stripped += 1
    # --- 重置求解器为快速默认值 ---
    spec.option.integrator = 0    # Euler
    spec.option.cone = 0          # pyramidal
    spec.option.impratio = 10.0
    spec.option.tolerance = 1e-8
    spec.option.iterations = 100

    if arm == "both":
        arm_distal = set(_arm_distal_links("left")) | set(_arm_distal_links("right"))
    else:
        arm_distal = set(_arm_distal_links(arm))
    body_links = set(BODY_COLLISION_LINKS)
    n_arm = 0
    n_body = 0
    waist_body = None
    for g in list(spec.geoms):
        if int(g.group) == 1:
            continue
        parent = g.parent
        bname = parent.name if parent is not None else ""
        if bname == "waist_yaw_link":
            waist_body = parent
            spec.delete(g)
            continue
        if bname in body_links:
            g.contype = BODY_COLLISION_BIT
            g.conaffinity = ARM_COLLISION_BIT
            n_body += 1
        elif bname in arm_distal:
            g.contype = ARM_COLLISION_BIT
            g.conaffinity = BODY_COLLISION_BIT
            n_arm += 1
        else:
            g.contype = 0
            g.conaffinity = 0

    if waist_body is not None:
        colliders = [_TORSO_COLLIDER] + _INSTRUMENT_COLLIDERS.get(instrument, [])
        for c in colliders:
            _add_box_collider(waist_body, **c)
            n_body += 1

    # 离屏渲染（--record）需要足够大的 framebuffer
    vis = spec.visual
    vis.global_.offwidth = max(vis.global_.offwidth, 640)
    vis.global_.offheight = max(vis.global_.offheight, 360)

    model = spec.compile()
    tag = instrument or "bare"
    extra = ""
    if n_stripped:
        extra = " stripped=%d" % n_stripped
    print(
        "[collision-filter] instrument=%s arm=%s arm_geoms=%d body_geoms=%d%s"
        % (tag, arm, n_arm, n_body, extra)
    )
    return model


def _add_box_collider(
    body: "mujoco.MjsBody",
    *,
    name: str,
    pos: tuple[float, float, float],
    half_size: tuple[float, float, float],
) -> None:
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size[:] = list(half_size)
    g.pos[:] = list(pos)
    g.group = 3  # 独立 group，不与已有 visual/collision group 冲突
    g.contype = BODY_COLLISION_BIT
    g.conaffinity = ARM_COLLISION_BIT
    g.rgba = [0.5, 0.8, 0.5, 0.25]  # 半透明绿，便于可视调试


def _arm_distal_links(arm: str) -> tuple[str, ...]:
    core = (
        f"{arm}_shoulder_roll_link",
        f"{arm}_shoulder_yaw_link",
        f"{arm}_elbow_pitch_link",
        f"{arm}_wrist_yaw_link",
        f"{arm}_wrist_pitch_link",
        f"{arm}_wrist_roll_link",
    )
    # 左手五指各指段（guitar 模型仅左手成指，右手是整体假手 mesh 已随 wrist_roll_link）
    if arm == "left":
        fingers = tuple(
            f"left_{finger}_{seg}_link"
            for finger in ("thumb", "index", "middle", "ring", "pinky")
            for seg in ("mcp_roll", "mcp_pitch", "mcp", "dip", "tip")
        ) + (
            "left_palm_base_link",
            "left_thumb_mcp_roll_link",
            "left_thumb_mcp_pitch_link",
            "left_thumb_dip_link",
        )
        return core + fingers
    return core


class _VideoRecorder:
    """离屏渲染 + 逐帧写 MP4。用法：每个物理步调 capture_if_needed()，最后 close()。"""

    def __init__(
        self,
        m: "mujoco.MjModel",
        d: "mujoco.MjData",
        path: Path,
        fps: int = 20,
        width: int = 640,
        height: int = 360,
    ):
        import imageio  # noqa: delayed import

        self._renderer = mujoco.Renderer(m, height=height, width=width)
        self._writer = imageio.get_writer(
            str(path), fps=fps, codec="h264", macro_block_size=1,
        )
        self._d = d
        self._path = path
        self._fps = fps
        self._step_interval = max(1, round(1.0 / (fps * m.opt.timestep)))
        self._step_count = 0
        self._frame_count = 0
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.lookat[:] = [0.0, 0.0, 0.75]
        self._cam.distance = 2.2
        self._cam.azimuth = 150
        self._cam.elevation = -15

    def capture_if_needed(self) -> None:
        self._step_count += 1
        if self._step_count % self._step_interval == 0:
            self._renderer.update_scene(self._d, camera=self._cam)
            frame = self._renderer.render()
            self._writer.append_data(frame)
            self._frame_count += 1

    def close(self) -> None:
        self._writer.close()
        dur = self._frame_count / max(self._fps, 1)
        print("视频已保存: %s (%d 帧, %.1f s)" % (self._path, self._frame_count, dur))


@dataclass
class MujocoCalibConfig:
    model_xml: Path
    urdf_path: Path
    arm: str
    # PD for <motor> actuators (N·m scale per model)；kp 比原先提到 240 以压缩
    # 大目标下的跟踪滞后，kd 对应抬到约 0.65·√(kp) 的比例保持过阻尼。
    kp: float = 240.0
    kd: float = 22.0
    max_motor_torque: float = 72.0
    move_max_steps: int = 12000
    move_tol: float = 0.08
    settle_steps: int = 40
    # 标定大姿态到位：每步对 _q_des 的限速 (rad/s)，与 HardStopCalibrator 的 pose_speed_scale 相乘。
    # 放宽到 1.5 rad/s（约 85°/s），命令层限速；实际关节由 PD 跟，不会超过 kp/kd 能驱动的上限。
    pose_max_vel_rad_s: float = 1.5
    # 靠限位搜索的关节角速度 = plan.search_velocity * 此系数
    # 注意：过小会导致手臂在躯干/碰撞处久停，易误触发判停、零偏明显偏离真限位，仅建议可视化调试用低值
    search_velocity_scale: float = 1.0
    # 到位超时兜底阻尼步数（不再做 qpos 直写，此参数只用于命令超时后的物理阻尼收尾）
    pose_snap_blend_steps: int = 36
    pose_snap_damp_steps: int = 200
    # 模拟原始编码器读数偏差（rad），标定后应估计出约 -bias（sign=+1 时）
    encoder_bias_rad: Dict[str, float] = field(default_factory=dict)
    # 贴近关节 range 时把判停力矩抬到至少该值，避免腕部等小关节传感器力矩偏低导致判停失败
    near_limit_torque_floor: float = 1.2
    # `mujoco.viewer.launch_passive` 返回的 Handle；非 None 时每步同步画面
    viewer_handle: Optional[Any] = None
    # 每多少步物理仿真调用一次 viewer.sync()，避免过慢（数值越小画面越流畅）
    viewer_sync_every: int = 4
    # 离屏录像器；非 None 时每步调 capture_if_needed()
    recorder: Optional[_VideoRecorder] = None


class MujocoArmCalibrationHardware:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, cfg: MujocoCalibConfig, joint_limits: Dict[str, JointLimit], arm_joints: tuple[str, ...], *, reset_data: bool = True):
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

        if reset_data:
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
        rec = self.cfg.recorder
        if rec is not None:
            rec.capture_if_needed()
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

        # 根据命令起点到目标的最远距离估算所需步数：ramp 到目标 + PD 跟踪余量 +
        # 配置的 move_max_steps 取三者最大。历史固定 12000 步在低速 v_max 下常常
        # 不够用（例如 0.049 rad/s × 12000 × 0.002s ≈ 1.18 rad），会回落到 qpos 直写
        # fallback，在 viewer 上表现为"瞬移 + 穿模"。
        max_dist = 0.0
        for jn, qg in q_goal.items():
            if jn in self._q_des:
                max_dist = max(max_dist, abs(qg - self._q_des[jn]))
        ramp_steps = int(max_dist / max(v_max * dt, 1e-9))
        pd_margin_steps = int(2.0 / max(dt, 1e-9))  # 给 PD ≈2s 跟踪余量
        max_steps = max(int(self.cfg.move_max_steps), ramp_steps + pd_margin_steps)

        for _ in range(max_steps):
            for jn, qg in q_goal.items():
                if jn not in self._qpos_adr:
                    continue
                e = qg - self._q_des[jn]
                self._q_des[jn] += float(np.clip(e, -v_max * dt, v_max * dt))
            self._step_physics()
            if self._converged_pose(q_goal, self.cfg.move_tol):
                return

        # 未收敛通常意味着命令目标本身被身体/关节 range 卡住；这里绝不对 qpos 直
        # 写，以免瞬移穿模。仅做额外阻尼让系统在真实物理下稳态收尾，即便仍超容
        # 差也不抛异常——标定循环紧接着会以当前位姿开始力矩顶靠，结果仍然可用。
        for _ in range(int(self.cfg.pose_snap_damp_steps)):
            self._step_physics()
            if self._converged_pose(q_goal, self.cfg.move_tol):
                return
        worst = 0.0
        worst_jn = ""
        for jn, qg in q_goal.items():
            if jn in self._qpos_adr:
                e = abs(self._read_joint_q(jn) - qg)
                if e > worst:
                    worst = e
                    worst_jn = jn
        if worst > self.cfg.move_tol * 2.0:
            print(
                "[move_to_pose] 未能精确到位 (最远 %s: %.3f rad)，按当前物理姿态继续" % (worst_jn, worst)
            )

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


def _arm_rest_pose(arm: str) -> Dict[str, float]:
    """双臂下垂但 roll 外展 10° 避开大腿的 rest pose。"""
    sign = 1.0 if arm == "left" else -1.0
    rest: Dict[str, float] = {}
    for j in arm_joint_names(arm):
        rest[j] = 0.0
    rest[f"{arm}_shoulder_roll_joint"] = sign * _REST_ROLL_RAD
    return rest


def _set_neutral_and_settle(
    plan,
    hardware: MujocoArmCalibrationHardware,
    arm: str,
    speed_scale: float,
) -> None:
    """先到中性/展臂等初始姿态，使用与标定相同的速度上限，避免开局瞬间甩臂穿模。

    当有乐器时手臂 neutral 需要很高的 shoulder_pitch (如 -2.5) 和大 roll (如 1.2)，
    直接一步到位会让前臂扫过乐器。waypoint 策略："先展后抬"——
      WP1: 先把 roll 展到 neutral 值（手臂外展，清开乐器琴身），pitch/elbow 保持 rest
      WP2: 再把 pitch 抬到 neutral 值（手臂已外展，避开琴身和琴颈）
      最后: 移动到完整 neutral（含 elbow 弯曲等）
    """
    neutral: Dict[str, float] = dict(plan.neutral_pose)
    rest = _arm_rest_pose(arm)
    other_arm = "left" if arm == "right" else "right"
    other_rest = _arm_rest_pose(other_arm)
    base: Dict[str, float] = {}
    for j in arm_joint_names(other_arm):
        if j in hardware._q_des:
            base[j] = other_rest.get(j, 0.0)
    for name in ("waist_yaw_joint", "head_yaw_joint", "head_pitch_joint"):
        if name in hardware._q_des:
            base[name] = 0.0

    sp_key = f"{arm}_shoulder_pitch_joint"
    sr_key = f"{arm}_shoulder_roll_joint"
    ep_key = f"{arm}_elbow_pitch_joint"
    target_pitch = neutral.get(sp_key, 0.0)
    target_roll = neutral.get(sr_key, 0.0)
    needs_waypoint = abs(target_pitch) > 1.5

    # 先到 rest pose（下垂外展 10°，避开大腿）
    rest_full = dict(base)
    rest_full.update(rest)
    hardware.move_to_pose(rest_full, speed_scale=speed_scale)
    for _ in range(max(hardware.cfg.settle_steps, 20)):
        hardware._step_physics()

    if needs_waypoint:
        # WP1: 展 roll 到 neutral 值，pitch/elbow 保持 rest
        wp1 = dict(base)
        wp1.update(neutral)
        wp1[sp_key] = 0.0
        wp1[ep_key] = 0.0
        wp1[sr_key] = target_roll
        hardware.move_to_pose(wp1, speed_scale=speed_scale)
        for _ in range(max(hardware.cfg.settle_steps, 30)):
            hardware._step_physics()

        # WP2: 抬 pitch（手臂已外展，安全绕过乐器），elbow 仍为 0
        wp2 = dict(base)
        wp2.update(neutral)
        wp2[ep_key] = 0.0
        hardware.move_to_pose(wp2, speed_scale=speed_scale)
        for _ in range(max(hardware.cfg.settle_steps, 30)):
            hardware._step_physics()

    pose = dict(base)
    pose.update(neutral)
    hardware.move_to_pose(pose, speed_scale=speed_scale)
    for _ in range(max(hardware.cfg.settle_steps, 60)):
        hardware._step_physics()


def _reset_arm(
    hw: MujocoArmCalibrationHardware,
    plan,
    arm: str,
    instrument: str,
    pose_speed_scale: float,
) -> None:
    """沿 setup 的逆序 waypoint 安全回到 rest pose（下垂外展 10°，避开大腿）。"""
    neutral = dict(plan.neutral_pose)
    rest = _arm_rest_pose(arm)
    sp_key = f"{arm}_shoulder_pitch_joint"
    sr_key = f"{arm}_shoulder_roll_joint"
    needs_waypoint = instrument and abs(neutral.get(sp_key, 0.0)) > 1.5

    if needs_waypoint:
        target_roll = neutral.get(sr_key, 0.0)
        # 1) 回到 neutral（从最后一步的任意 hold pose 回到已知安全姿态）
        hw.move_to_pose(neutral, speed_scale=pose_speed_scale)
        # 2) pitch 降到 0，roll 保持外展
        rw2 = dict(rest)
        rw2[sr_key] = target_roll
        hw.move_to_pose(rw2, speed_scale=pose_speed_scale)
        # 3) roll 收回 rest 值
        hw.move_to_pose(rest, speed_scale=pose_speed_scale)
    else:
        hw.move_to_pose(rest, speed_scale=pose_speed_scale)
    hw.sleep(0.3)
    print("[%s] 手臂已复位" % arm)


def run_calibration(
    model_xml: Path,
    urdf_path: Path,
    arm: str,
    out_yaml: Path,
    encoder_bias: Optional[Dict[str, float]] = None,
    *,
    visualize: bool = False,
    record_path: Optional[Path] = None,
    record_fps: int = 30,
    viewer_sync_every: int = 4,
    pose_max_vel_rad_s: float = 1.5,
    search_velocity_scale: float = 1.0,
    settle_seconds: float = 0.06,
    pose_speed_scale: float = 0.8,
    search_mode: str = "torque_damping",
    torque_search_nm: float = 8.0,
    torque_damping_nm_s: float = 3.0,
) -> Dict[str, float]:
    spec = mujoco.MjSpec.from_file(str(model_xml.resolve()))
    instrument = _detect_instrument(spec)
    del spec
    # keyboard 虽然琴体已剥离，但实机上电子琴在身前，采用 guitar 的高抬臂轨迹
    _PLAN_INSTRUMENT_MAP = {"keyboard": "guitar"}
    plan_instrument = _PLAN_INSTRUMENT_MAP.get(instrument, instrument if instrument in _CONTACT_INSTRUMENTS else "")
    arms = ["left", "right"] if arm == "both" else [arm]
    m = load_model_with_collision_filter(model_xml, "both" if arm == "both" else arm)
    d = mujoco.MjData(m)
    recorder: Optional[_VideoRecorder] = None
    if record_path is not None:
        recorder = _VideoRecorder(m, d, record_path, fps=record_fps)

    def _calibrate_one_arm(
        side: str, viewer: Optional[Any], *, reset_data: bool = True
    ) -> Dict[str, float]:
        all_limits = parse_joint_limits(urdf_path)
        arm_list = arm_joint_names(side)
        joint_limits = {n: all_limits[n] for n in arm_list}
        cfg = MujocoCalibConfig(
            model_xml=model_xml,
            urdf_path=urdf_path,
            arm=side,
            encoder_bias_rad=encoder_bias or {},
            viewer_handle=viewer,
            viewer_sync_every=viewer_sync_every,
            pose_max_vel_rad_s=pose_max_vel_rad_s,
            search_velocity_scale=search_velocity_scale,
            recorder=recorder,
        )
        plan = build_default_arm_calibration_plan(urdf_path, side, instrument=plan_instrument)
        hw = MujocoArmCalibrationHardware(m, d, cfg, joint_limits, arm_list, reset_data=reset_data)
        _set_neutral_and_settle(plan, hw, side, speed_scale=pose_speed_scale)
        det = HardStopDetectorConfig(
            stall_time_seconds=0.12,
            position_window_epsilon=0.015,
            velocity_epsilon=0.2,
            min_current_ratio=0.08,
            sample_timeout_seconds=30.0,
            backoff_seconds=0.08,
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
        print("MuJoCo 标定 [%s]: arm=%s, dt=%.4f" % (mode, side, m.opt.timestep))
        hw.sleep(0.2)
        offsets = calib.calibrate(plan, cast(CalibrationHardware, hw), persist=False)
        print("[%s] 标定完成:" % side)
        for k, v in sorted(offsets.items()):
            b = cfg.encoder_bias_rad.get(k, 0.0)
            if abs(b) > 1e-9:
                print("  %s  offset=%.6f  (注入 bias=%.6f, 预期 offset≈%.6f)" % (k, v, b, -b if signs[k] == 1.0 else b))
            else:
                print("  %s  offset=%.6f" % (k, v))
        _reset_arm(hw, plan, side, plan_instrument, pose_speed_scale)
        return offsets

    def _execute_all(viewer: Optional[Any]) -> Dict[str, float]:
        all_offsets: Dict[str, float] = {}
        for i, side in enumerate(arms):
            all_offsets.update(_calibrate_one_arm(side, viewer, reset_data=(i == 0)))
        if recorder is not None:
            recorder.close()
        write_zero_offsets_yaml(
            out_yaml,
            all_offsets,
            header_lines=(
                "MuJoCo hard-stop zero-offset calibration (radians).",
                "model=%s arm=%s" % (model_xml.name, arm),
            ),
        )
        print("已写入: %s" % out_yaml)
        return all_offsets

    if visualize:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            result = _execute_all(viewer)
            print("标定完成，关闭 MuJoCo 窗口以退出…")
            import time as _time
            while viewer.is_running():
                _time.sleep(0.05)
            _time.sleep(0.2)
            return result
    return _execute_all(None)


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
    ap.add_argument("--arm", choices=("left", "right", "both"), required=True)
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
        "--record",
        type=Path,
        metavar="VIDEO.mp4",
        help="将标定过程离屏渲染并录制为 MP4 视频（无需 --visualize，可同时使用）",
    )
    ap.add_argument(
        "--record-fps",
        type=int,
        default=20,
        help="录像帧率（默认 20）",
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
        default=1.5,
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
        default=0.06,
        help="每步标定中到位后等待时间（仿真秒）",
    )
    ap.add_argument(
        "--pose-speed-scale",
        type=float,
        default=0.8,
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
        sides = ["left", "right"] if args.arm == "both" else [args.arm]
        for s in sides:
            print(plan_summary(build_default_arm_calibration_plan(args.urdf, s)))
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
        record_path=args.record,
        record_fps=max(1, int(args.record_fps)),
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
