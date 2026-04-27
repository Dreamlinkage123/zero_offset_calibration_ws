## ROS 2 功能包布局

本目录已封装为 ament_python 包 `zero_offset_calibration`（`src/zero_offset_calibration/`）：

```
zero_offset_calibration/
├── package.xml / setup.py / setup.cfg
├── resource/zero_offset_calibration       # ament 资源标记
├── zero_offset_calibration/               # Python import 路径
│   ├── __init__.py
│   ├── _paths.py                           # share/ 目录定位辅助
│   ├── hard_stop_calibration.py            # 规划 + 运行骨架（无 ROS/MuJoCo）
│   ├── mujoco_hard_stop_calibration.py     # MuJoCo 仿真适配器
│   └── ros2_upper_body_hardware.py         # ROS 2 真机适配器
├── casbot_band_urdf/                       # 安装到 share/<pkg>/casbot_band_urdf/
│   └── urdf/ xml/ meshes/
├── config/                                 # 示例零偏 YAML
├── launch/ros2_upper_body_hardware.launch.py
└── test/test_hard_stop_calibration.py
```

配套 `src/crb_ros_msg/` 提供 `UpperJointData.msg`，是本包的一般依赖
（`depend` 于 `package.xml`）。

### 交叉编译（docker_env）

工作区根目录的 `docker_env/build.sh` 对 `colcon build --merge-install` 做了封装：
无顶层 `CMakeLists.txt` 时自动按 colcon 工作区处理。x86 开发机：

```bash
./docker_env/docker_load_x86_aarch64.sh x86
./docker_env/build.sh x86
```

aarch64（目标机，Jetson/Orin）：

```bash
./docker_env/docker_load_x86_aarch64.sh aarch64
./docker_env/build.sh aarch64
```

产物分别在 `install_x86/`、`install_aarch64/`，同名 `install/` 符号链接指向
当前架构版本。

#### 架构差异：MuJoCo 仿真仅限 x86

| 功能 | x86 | aarch64 (Orin) |
|---|---|---|
| `hard_stop_calibration`（纯规划） | 可用 | 可用 |
| `ros2_upper_body_hardware`（真机标定） | 可用 | 可用 |
| `calibration_web_ui`（Web 控制台） | 可用 | 可用 |
| `mujoco_hard_stop_calibration`（MuJoCo 仿真） | 可用 | **不可用** |

- `mujoco` 和 `numpy` 仅在 x86 开发机上安装；aarch64 的 Docker 镜像不包含这两个库。
- `mujoco_hard_stop_calibration.py` 使用**延迟导入**：模块可以被 `colcon build` 正常打包和安装，
  但在 aarch64 上执行 `ros2 run zero_offset_calibration mujoco_hard_stop_calibration` 时会立刻给出
  明确的错误提示，并建议使用真机标定入口。
- 其余入口（`hard_stop_calibration`、`ros2_upper_body_hardware`、`calibration_web_ui`）
  无 MuJoCo/numpy 依赖，两个架构上均正常工作。

### 运行入口（ros2 run）

```bash
source install/setup.bash   # 或 install_x86/setup.bash / install_aarch64/setup.bash

# 1) 仅打印左臂校准计划（纯规划，无 ROS/MuJoCo 依赖）
ros2 run zero_offset_calibration hard_stop_calibration --arm left --print-plan \
    --urdf $(ros2 pkg prefix zero_offset_calibration)/share/zero_offset_calibration/casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf

# 2) MuJoCo 仿真标定（仅 x86；需 mujoco、numpy，aarch64 上不可用）
ros2 run zero_offset_calibration mujoco_hard_stop_calibration --arm both \
    --out /tmp/zero_offsets_mujoco.yaml

# 3) ROS 2 真机标定 — bass 乐器（默认）
ros2 run zero_offset_calibration ros2_upper_body_hardware --arm right --persist \
    --skip-on-timeout

# 4) ROS 2 真机标定 — guitar 乐器（--instrument 同时决定 URDF 和避障姿态）
ros2 run zero_offset_calibration ros2_upper_body_hardware --arm right --persist \
    --skip-on-timeout --instrument guitar

# 5) 或用 launch
ros2 launch zero_offset_calibration ros2_upper_body_hardware.launch.py \
    arm:=right persist:=true skip_on_timeout:=true instrument:=bass
```

#### `--instrument` 参数

`--instrument` 同时决定**默认加载的 URDF** 和**标定过程中的避障姿态**：

| `--instrument` | 默认 URDF | 避障姿态 |
|---|---|---|
| **bass**（默认） | `…_P1L_bass.urdf` | bass 高抬臂 |
| **guitar** | `…_P1L_guitar.urdf` | guitar 高抬臂 |
| auto | 沿用 `--urdf` 指定的文件名自动识别 | 自动 |
| none | bass URDF | 裸机（不避障） |
| keyboard | bass URDF | guitar 高抬臂 |

显式传 `--urdf <path>` 会覆盖 `--instrument` 的自动 URDF 选择。

#### `--skip-on-timeout`

启用后，某个关节在标定过程中任一阶段（`move_to_pose` 到位、搜索超时、卡滞早停）
发生 `TimeoutError` 时，该关节被**跳过**并继续下一个关节，而不会中止整臂标定。
跑完后日志最终会 `WARN` 列出被跳过的关节，成功标定的关节仍正常写入 YAML。

#### 判停参数默认值

真机 ROS 2 入口的默认判停阈值较仿真放宽（velocity_epsilon 0.08、
position_window 0.012 rad、stall_time 0.50 s、超时 20 s、min_current_ratio 0.10），
全部可通过 CLI 覆盖；搜索循环每 2 s 打印一行诊断，直观显示哪个条件未达标。

---

# 机器人手臂硬限位零偏校准说明

本仓库原先只有 `URDF/XML` 资产，没有控制或校准程序。现在新增了 `hard_stop_calibration.py`，用于两件事：

1. 从 `URDF` 提取左右臂 7 个关节的转轴、限位和默认校准顺序。
2. 提供一套"去硬限位 -> 检测触碰 -> 计算零偏 -> 回退卸力"的运行骨架，等待你接入底层电机/编码器接口。

## 适用关节链

左右臂各自按 7 自由度处理：

- `*_shoulder_pitch_joint`
- `*_shoulder_roll_joint`
- `*_shoulder_yaw_joint`
- `*_elbow_pitch_joint`
- `*_wrist_yaw_joint`
- `*_wrist_pitch_joint`
- `*_wrist_roll_joint`

支持四种模型（关节链完全一致，区别在于乐器附件）：

| 模型 | XML 文件 | URDF 文件 | 乐器附件 |
|------|---------|----------|---------|
| bare（纯躯干） | `*_P1L.xml` | `*_P1L.urdf` | 无 |
| guitar | `*_P1L_guitar.xml` | `*_P1L_guitar.urdf` | 吉他（STL 合并在 waist mesh 中） |
| bass | `*_P1L_bass.xml` | `*_P1L_bass.urdf` | 贝斯（STL 合并在 waist mesh 中） |
| keyboard | `*_P1L_keyboard.xml` | `*_P1L_keyboard.urdf` | 电子琴（独立 body，含琴键铰链） |

## 零偏标定原理

### 问题定义

工业机器人在组装、维修或更换编码器后，编码器零点与关节物理零位之间存在一个固定偏差，称为**零位偏移（zero offset）**。如不补偿，机器人运动学解算和轨迹规划会产生系统性误差。

硬限位零偏标定利用关节运动范围两端的**机械硬停**（hard stop）作为已知参考点：将关节缓慢驱动至机械硬限位，通过检测运动停止来确定编码器在该位置的读数，再与模型中标注的硬限位角度对比，即可算出偏移量。

### 数学模型

关节角与编码器的对应关系为：

```
θ = s · e + δ
```

| 符号 | 含义 |
|------|------|
| θ | 真实关节角度（rad），定义于 URDF 坐标系 |
| e | 编码器原始读数（rad） |
| s | 编码器符号（`+1` 或 `-1`），取决于编码器安装方向与 URDF 正方向的关系 |
| δ | 零位偏移（zero offset），即要求解的未知量 |

当关节顶到机械硬限位时，真实角度 θ 等于 URDF/XML 中定义的关节限位值 `θ_stop`，此时编码器读数为 `e_stop`，代入得：

```
θ_stop = s · e_stop + δ
```

解出：

```
δ = θ_stop − s · e_stop
```

对应代码 `HardStopCalibrator.compute_zero_offset()`：

```python
offset = reference_angle - encoder_sign * encoder_position
```

### URDF 关节限位（实测值）

`CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf` 中左右臂 14 个关节的限位已按真机实测值更新，左右臂对称：

| 关节 | lower (rad) | upper (rad) |
|------|-------------|-------------|
| `*_shoulder_pitch_joint` | -3.22886 | 1.60570 |
| `left_shoulder_roll_joint` | -0.34906585 | 3.14159265 |
| `right_shoulder_roll_joint` | -3.14159265 | 0.34906585 |
| `*_shoulder_yaw_joint` | -1.57079633 | 1.57079633 |
| `*_elbow_pitch_joint` | -2.26893 | 0.34907 |
| `*_wrist_yaw_joint` | -1.57079633 | 1.57079633 |
| `*_wrist_pitch_joint` | -1.04719755 | 1.04719755 |
| `left_wrist_roll_joint` | -1.57079633 | 1.04719755 |
| `right_wrist_roll_joint` | -1.04719755 | 1.57079633 |

> **注意**：`elbow_pitch` 的 `upper=0.34907` 需要驱动软限位已放开至该值。如果驱动侧
> 仍锁在旧限位（如 upper≈0），`move_to_pose` 会因无法到达 approach 角度而超时。

### 前提假设

1. **硬限位与模型一致**：物理硬停位置必须与 URDF `<limit lower="..." upper="..."/>` 中的值吻合。若存在缓冲垫、弹性止挡等软因素，需要在 `stop_angle` 中补偿。
2. **编码器线性且无间隙**：编码器 e 与关节角 θ 之间仅差一个符号和常数偏移，没有非线性或多圈漂移。
3. **符号已知**：`encoder_sign` 必须事先确认；如果 s 搞反，offset 的误差为 2·s·e_stop，通常远大于正常偏移值。

### 单关节标定流程

对每个待标定关节，执行以下 5 步：

```
┌──────────────────────────────────────────────────────────┐
│ ① hold_pose: 将所有关节移到安全姿态，目标关节到 approach  │
│ ② settle:    等待惯性衰减（默认 0.06 s）                 │
│ ③ search:    对目标关节施加恒力矩（或恒速度），驱向硬限位 │
│ ④ detect:    滑动窗口判停，确认已顶到硬限位              │
│ ⑤ compute:   读取编码器，计算 δ = θ_stop − s · e_stop    │
│ ⑥ backoff:   回退到安全角度，解除关节力                   │
└──────────────────────────────────────────────────────────┘
```

#### ① hold_pose — 安全预置

先将 7 个关节整体移到一个"holding pose"，使手臂远离身体和乐器，同时把目标关节定位到距硬限位约 0.20 rad 的 approach 角度。其余关节保持 neutral/外展姿态，避免搜索过程中前臂或腕部扫过障碍物。

holding pose 因目标关节不同而变化——例如标定 shoulder_roll 时 shoulder_pitch 更低（手臂下垂更多），标定 wrist 时 elbow 更弯（缩短前臂力臂）。详见 `step_hold_pose()` 的分支逻辑。

#### ② settle — 惯性衰减

移动完成后等待一小段时间（`settle_seconds`），让关节振荡衰减至稳态，确保后续采样不受过渡态干扰。

#### ③ search — 驱动到硬限位

支持两种搜索模式：

| 模式 | 控制律 | 适用场景 |
|------|--------|---------|
| `torque_damping`（默认） | τ = sign · T − b · q̇ | 仿真与力矩可控的驱动器，接触刚性更好 |
| `velocity` | q̇_cmd = search_velocity | 仅支持速度/位置接口的驱动器 |

- **力矩-阻尼模式**：施加固定幅值的恒力矩 T（默认 8 N·m），同时加上与角速度成比例的阻尼项 b（默认 3 N·m·s/rad），使关节缓慢、平稳地顶靠到硬限位。阻尼项让关节在硬限位附近不会产生冲击。
- **速度模式**：直接给一个低速命令（由 `search_velocity` 确定），在速度环层面驱动关节；适合无法下发力矩的位置/速度接口。

#### ④ detect — 滑动窗口判停

硬限位检测使用一个**固定时间窗口**（`stall_time_seconds`）内的采样历史，需要满足**运动学条件**（全部 AND）加上**到位确认**条件才判定已顶到硬停：

```
判停条件 = 时间充足 ∧ 位置稳定 ∧ 速度为零 ∧ 电流达标 ∧ 行程足够
           ∧ (几何近邻 ∨ 动态力矩增量)      ← 两条到位证据任一通过即可
```

**运动学条件**（必须全部满足）：

| 条件 | 公式 | 仿真默认 | 真机默认 | 含义 |
|------|------|---------|---------|------|
| 时间充足 | t_newest − t_oldest ≥ stall_time × 0.98 | 0.20 s | 0.50 s | 窗口已收集了足够长的数据 |
| 位置稳定 | \|e_newest − e_oldest\| ≤ ε_pos | 0.003 rad | 0.012 rad | 编码器在窗口内几乎没动 |
| 速度为零 | \|v_newest\| ≤ ε_vel | 0.015 rad/s | 0.08 rad/s | 当前估计角速度接近零 |
| 电流达标 | \|I_newest\| ≥ I_threshold × ratio | ratio=0.30 | ratio=0.10 | 电机电流表明关节在承受负载；阈值≤0 时跳过 |
| 行程足够 | \|pos − start\| ≥ min_search_travel | 0.0 (关闭) | 0.08 rad | 防止还没推动关节就被判停 |

**到位确认条件**（两条都开启时为 OR——任一通过即可）：

| 条件 | 公式 | 默认阈值 | 含义 |
|------|------|---------|------|
| 几何近邻 | \|sign × enc − stop_angle\| ≤ max_expected_offset | 0.60 rad | 判停位置在 stop_angle 附近 |
| 动态力矩增量 | \|effort\| ≥ baseline + effort_rise_nm | 0.15 Nm | 力矩相比自由运动上升（仅 current_threshold≤0 时启用） |

几何近邻和动态力矩增量采用 **OR** 逻辑：在 torque-damping 搜索模式下，撞到限位后驱动力矩可能回落到低于自由运动基线，effort_rise 条件不成立，但几何上已经紧贴 stop_angle（dist2stop < 10 mrad），此时仅靠几何近邻即可确认到位。反之，如果 URDF stop_angle 标注偏差较大导致几何超标，但力矩增量明确，也接受。

滑动窗口的实现：每次采样后剔除窗口之外的旧样本，保留最近 `stall_time_seconds` 内的数据。这比固定采样数更鲁棒——不依赖采样周期。

对应代码：

```python
# 采样循环（hard_stop_calibration.py: HardStopCalibrator.calibrate）
while True:
    sample = hardware.read_sample(step.target_joint)
    history.append(sample)
    # 只保留 stall_time 时间窗口内的采样
    history = [h for h in history
               if sample.timestamp - h.timestamp <= stall_time]

    if self._stopped_on_hard_limit(history, current_threshold):
        # 检测到硬限位，计算零偏
        offset = reference_angle - sign * sample.encoder_position
        break
```

#### ⑤ compute — 计算零偏

判停后立即读取当前编码器值 `e_stop`，用上面推导的公式计算：

```
δ = θ_stop − s · e_stop
```

计算完毕后调用 `apply_zero_offset()` 将偏移写入控制器 RAM，**后续标定步骤立即生效**。

#### ⑥ backoff — 卸力回退

将目标关节回退到距硬限位约 0.06 rad 的安全位置（`backoff_angle`），避免关节长期顶在硬限位上造成电机过热或机械磨损。

### 全臂标定编排

单臂 7 个关节按固定顺序逐一标定（默认：roll → pitch → yaw → elbow → 3×wrist），每个关节执行上述 ①–⑥ 完整流程。顺序的设计原则：

1. **先大关节后小关节**：shoulder 先标定，建立近端精度基准；wrist 最后标定，此时肩肘已经校准。
2. **roll 先于 pitch**：roll 校准后手臂展开方向准确，pitch 搜索方向不受 roll 偏差影响。
3. **安全递进**：每个关节在 hold_pose 中对其余 6 个关节都有明确约束，确保标定运动不穿越障碍。

### 关键参数及其物理意义

| 参数 | 仿真默认 | 真机默认 | 作用 | 调整建议 |
|------|---------|---------|------|---------|
| `stop_angle` | URDF 上/下限 | URDF 上/下限（已按实测更新） | 硬限位参考角度 | 若有缓冲垫需修正 |
| `approach_angle` | stop − 0.20 rad | 同左 | 搜索起始位置 | 不宜太远（浪费时间）也不宜太近（可能误判） |
| `backoff_angle` | stop − 0.06 rad | 同左 | 卸力位置 | 大于碰撞体回弹距离即可 |
| `search_velocity` | ±0.05–0.20 rad/s | 同左 | 速度模式搜索速度 | 太快容易冲过，太慢浪费时间 |
| `torque_search_nm` | 8.0 N·m | 8.0 N·m | 力矩模式恒力矩 | 需大于关节摩擦+重力矩，但不可过大 |
| `torque_damping_nm_s` | 3.0 N·m·s/rad | 3.0 N·m·s/rad | 力矩模式阻尼系数 | 越大越柔，但响应越慢 |
| `stall_time_seconds` | 0.20 s | 0.50 s | 判停窗口长度 | 增大提高稳定性但增加时间 |
| `position_window_epsilon` | 0.003 rad | 0.012 rad | 窗口内位置变化阈值 | 编码器噪声大则增大 |
| `velocity_epsilon` | 0.015 rad/s | 0.08 rad/s | 速度判零阈值 | 与估算精度匹配 |
| `min_current_ratio` | 0.30 | 0.10 | 电流达标比例 | 空载电流大则增大 |
| `min_search_travel` | 0.0 (关闭) | 0.08 rad | 最小搜索行程 | 防止 approach 点直接误判 |
| `max_expected_offset` | 0.0 (关闭) | 0.60 rad | 几何近邻门限 | 手腕等机械回差大的关节可适当调大 |
| `effort_rise_nm` | 0.0 (关闭) | 0.15 Nm | 动态力矩增量门限 | 仅 current_threshold≤0 时启用 |
| `stuck_abort_seconds` | 0.0 (关闭) | 4.0 s | 卡滞早停时长 | 卡在中途时比超时更快失败 |
| `sample_timeout_seconds` | 10.0 s | 20.0 s | 单关节搜索超时 | 配合 `--skip-on-timeout` 使用 |
| `current_thresholds` | 0.35 A | 5.0 / 0.0 (腕) | 各关节电流门限 | 实机需实测；腕关节默认 0.0 禁用绝对电流判据 |
| `encoder_signs` | +1 | +1 | 编码器方向符号 | 必须根据实机确认 |
| `move_timeout_s` | — | 15.0 s | `move_to_pose` 总超时 | 配合 stuck 检测使用 |
| `move_stuck_window_s` | — | 3.0 s | `move_to_pose` 无进度窗口 | 窗口内位置变化 < epsilon 即判定卡死 |
| `move_stuck_epsilon` | — | 0.010 rad | `move_to_pose` 无进度门限 | — |

### 仿真与真机的差异

| 方面 | MuJoCo 仿真 | 真机（ROS2） |
|------|------------|-------------|
| 硬限位来源 | `<joint range="..."/>` | 物理硬停 |
| 编码器 | `d.qpos[joint_adr]`（无噪声） | `JointState.position`（有噪声） |
| 电流 | `d.sensordata[tau_adr]`（理想力矩） | `JointState.effort`（含摩擦等） |
| 时间戳 | 仿真时间（可快于实时） | `time.monotonic()`（实时） |
| 搜索方式 | 力矩-阻尼（直接控制 ctrl） | 位置增量（UpperJointData） |
| shoulder_roll offset | ≈±0.03 rad（碰撞体误差） | 取决于实际硬停 |

## 默认姿态设计原则

默认姿态不是碰撞证明后的最优解，而是依据 `URDF` 结构生成的推荐值，原则如下：

- 肩关节优先把手臂外展，尽量远离躯干；
- 肘部默认保持一定弯曲，减少前臂扫到本体的概率；
- 腕部校准时，肩肘保持在稳定的中间姿态，只让末端关节单独搜索；
- 左右臂对称关节会选择不同的推荐硬限位方向，例如左右肩 roll、左右腕 roll；
- `shoulder_roll` 选"贴身"窄端（`±0.3491 rad`）——URDF 里该关节两侧极度不对称，一侧 `±π`，从悬垂位绕过头顶必然先撞到躯干/头部，仿真中会导致零偏算错、动作穿模（详见 `hard_stop_calibration.preferred_stop_side` 注释）。

### Rest pose（避开大腿）

双臂自然下垂（全零位）时前臂会与大腿干涉。仿真中定义了 rest pose：`shoulder_roll` 外展 10°（0.175 rad），其余关节为 0，作为标定的起始和结束姿态。常量 `_REST_ROLL_RAD` 在 `mujoco_hard_stop_calibration.py` 中，可按需调整。

### 乐器感知的避障姿态

当 `build_default_arm_calibration_plan()` 传入 `instrument="guitar"` 或 `"bass"` 时，neutral 和 hold pose 会自动调整以避开乐器碰撞体：

| 参数 | 无乐器 | 有乐器 |
|------|--------|--------|
| neutral shoulder_pitch | -1.10 rad | **-2.50 rad**（手臂高举过头，前臂从上方绕过琴颈） |
| neutral shoulder_roll | ±0.60 rad | **±1.20 rad**（更大外展，远离琴身） |
| neutral elbow_pitch | -0.90 rad | **-0.50 rad**（放松弯曲，减小前臂横向伸展） |
| shoulder_roll hold pitch | -1.20 rad | **-2.50 rad**（保证 roll 内收至 -0.35 时前臂在琴颈上方） |
| 其它 hold pitch | -0.95 ~ -1.00 | **-2.50 rad** |

### Waypoint 轨迹规划

MuJoCo 仿真中大幅度姿态过渡采用分段 waypoint 策略，避免同步移动多个关节时前臂扫过乐器：

**标定开始（rest → neutral）：**

1. 先到 rest pose（双臂外展 10° 避开大腿）
2. WP1："先展"——roll 展到 neutral 值，pitch/elbow 保持 0（手臂在外侧，远离琴身）
3. WP2："后抬"——pitch 抬到 neutral 值（手臂已外展，安全绕过琴身和琴颈）
4. 到完整 neutral（含 elbow 弯曲等）

**标定结束（neutral → rest），setup 的精确逆序：**

1. 回到 neutral（从最后一步 hold pose 回到已知安全姿态）
2. rev-WP2：pitch 降到 0，roll 保持外展（手臂在外侧向下摆）
3. rev-WP1：roll 收回 rest 值（手臂已悬垂，收 roll 不碰乐器）

无乐器模型（bare）的 neutral pitch 较小，不需要 waypoint，直接从 rest 到 neutral。

## 推荐校准顺序

每只手臂默认顺序：

1. shoulder roll
2. shoulder pitch
3. shoulder yaw
4. elbow pitch
5. wrist yaw
6. wrist pitch
7. wrist roll

如果你们实际机构在某个顺序下更安全，可以直接改 `build_default_arm_calibration_plan()` 里的关节顺序或 `preferred_stop_side()` 的目标硬限位方向。

## 如何查看自动生成的校准计划

查看左臂计划：

```bash
python3 hard_stop_calibration.py \
  --urdf casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf \
  --arm left \
  --print-plan
```

查看双臂计划：

```bash
python3 mujoco_hard_stop_calibration.py --arm both --print-plan
```

输出会同时给出：

- 每个关节的上下限
- 推荐去的硬限位方向
- 搜索前的 approach 角度
- 顶到限位后的 backoff 角度
- 该步骤其余关节的 holding pose

## 需要你接入的底层接口

非 ROS 场景下，需要自行实现 `CalibrationHardware` 协议中的这些方法：

- `move_to_pose()`
- `read_sample()`
- `start_velocity_search()`
- `stop_joint()`
- `apply_zero_offset()`
- `persist_zero_offsets()`
- `sleep()`

还需要提供：

- `encoder_signs`：每个关节编码器方向
- `current_thresholds`：每个关节判定顶到硬限位的电流阈值

## CASBOT02 真机（《上半身》二次开发接口）

工程内文档：仓库根目录 `CASBOT02 二次开发文档-上半身.docx`。与上身零偏校准则直接相关的有：

- **3.2 上身关节控制**：服务 `/motion/upper_body_debug`（`std_srvs/srv/SetBool`）；话题 `/upper_body_debug/joint_cmd`（`crb_ros_msg` 的 `UpperJointData`）。
- **2.5.3 关节状态数据**：`sensor_msgs/msg/JointState`，topic **`/joint_states`**（含腿、腰、头、臂、灵巧手等关节的位姿与力矩），本仓库脚本默认以此作为反馈。
- **2.5.2 IMU 数据**（**`/imu`**，`sensor_msgs/msg/Imu`）：姿态用；**不参与**本硬限位零偏计算，需要时可自订阅。

实机已对接脚本：`ros2_upper_body_hardware.py`（在已 `source` ROS2 与厂商工作空间、且可 `import crb_ros_msg` 的环境中运行）。

**关节反馈**：与文档 2.5.3 一致，默认订阅 `/joint_states`。若现场有 remap 或其它命名，用 `--joint-states` 覆盖。

**顶靠方式**：`UpperJointData` 为**位置**接口。搜索线程在软限位内小步长逼近硬限位；与纯速度环顶靠是否等效，取决于现场位置跟踪。

**零偏落盘**：该节未提供零偏专用写入接口。`--persist` 时用 `write_zero_offsets_yaml()` 将测算的零偏写入 `zero_offsets.yaml`（**标准 YAML**，不依赖安装 PyYAML），供电驱或内部参数服务消费。

```bash
# 打印计划
ros2 run zero_offset_calibration ros2_upper_body_hardware --arm right --print-plan-only

# 真机标定（bass 乐器，默认）
ros2 run zero_offset_calibration ros2_upper_body_hardware --arm right --persist --skip-on-timeout

# 真机标定（guitar 乐器）
ros2 run zero_offset_calibration ros2_upper_body_hardware --arm right --persist --skip-on-timeout --instrument guitar
```

## MuJoCo 仿真标定

支持仓库内 `casbot_band_urdf/xml/` 下的四种模型（bare / guitar / bass / keyboard），自动检测乐器类型并生成对应碰撞 primitive 和避障姿态。支持 `--arm left`、`--arm right` 或 `--arm both`（双臂顺序标定，共用一个模型和 viewer 窗口）。

腿部无执行器，仿真中在每一步将腿关节锁回初值。`move_to_pose` 按"当前位姿→目标"的真实距离动态分配步数，不足时追加阻尼收尾，**不做 qpos 直写的"瞬移"回退**，避免 viewer 上看到的跳变和穿模。

依赖：`pip install mujoco`（本机已测 MuJoCo 3.3.x）。`read_sample` 使用**仿真时间戳**，与 `HardStopCalibrator` 配套；判停逻辑在 `hard_stop_calibration.py` 中对停留时长做了少量容差，避免固定步长导致永远达不到"满窗口时长"。

**标定流程（以 `--arm both` 为例）：**

1. 模型加载，双臂碰撞过滤同时启用
2. 左臂：rest → waypoint setup → neutral → 7 关节标定 → waypoint reset → rest
3. 右臂：rest → waypoint setup → neutral → 7 关节标定 → waypoint reset → rest
4. 写出合并的 14 关节 YAML
5. `--visualize` 时 viewer 保持打开，关闭窗口退出

```bash
# 双臂标定（默认 guitar 模型）
python3 mujoco_hard_stop_calibration.py --arm both --out zero_offsets_mujoco.yaml

# 双臂标定 + 可视化
python3 mujoco_hard_stop_calibration.py --arm both --out zero_offsets_mujoco.yaml --visualize

# 单臂标定
python3 mujoco_hard_stop_calibration.py --arm left --out zero_offsets_mujoco.yaml

# 仅打印 URDF 计划（双臂）
python3 mujoco_hard_stop_calibration.py --arm both --print-plan

# 使用 bass 模型
python3 mujoco_hard_stop_calibration.py \
  --model-xml casbot_band_urdf/xml/CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.xml \
  --urdf casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf \
  --arm both --out zero_offsets_mujoco.yaml

# 使用 bare（无乐器）模型
python3 mujoco_hard_stop_calibration.py \
  --model-xml casbot_band_urdf/xml/CASBOT02_ENCOS_7dof_shell_20251015_P1L.xml \
  --urdf casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L.urdf \
  --arm both --out zero_offsets_mujoco.yaml

# 使用 keyboard（电子琴）模型
python3 mujoco_hard_stop_calibration.py \
  --model-xml casbot_band_urdf/xml/CASBOT02_ENCOS_7dof_shell_20251015_P1L_keyboard.xml \
  --urdf casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_keyboard.urdf \
  --arm both --out zero_offsets_mujoco.yaml

# 打开 MuJoCo 交互窗口，实时看机械臂各步顶靠与姿态
python3 mujoco_hard_stop_calibration.py --arm both --out zero_offsets_mujoco.yaml --visualize

# 降低 sync 频率加快速度：每 8 步仿真再刷新一帧
python3 mujoco_hard_stop_calibration.py --arm both --visualize --viewer-sync-every 8

# 注入常值编码器偏差（JSON），用于验证 offset ≈ -bias（sign=+1）
# echo '{"right_wrist_yaw_joint": 0.05}' > /tmp/bias.json
# python3 mujoco_hard_stop_calibration.py --arm right --encoder-bias /tmp/bias.json
```

无图形环境（如 CI）下不要用 `--visualize`；本机需可用 OpenGL/GLFW 的显示环境。

说明：仿真里"硬限位"即关节 `range` 约束，与真机机械硬停有差别；`shoulder_roll` 因躯干碰撞体比 URDF limit 提前约 0.03 rad 拦住手臂，导致 offset 约 ±0.03 rad，其余 6 个关节 offset < 0.002 rad。

### 手臂 ↔ 躯干/乐器 碰撞过滤（`load_model_with_collision_filter`）

#### 根因：XML 默认禁用所有碰撞

原始 XML 的 `<default>` 里设了 `conaffinity="0"`，所有 body 的碰撞几何都没有显式覆盖，MuJoCo 的位掩码判定 `(contype_i & conaffinity_j) | (contype_j & conaffinity_i)` 永远为 0——**整个模型没有任何 body 间碰撞**。

#### 为什么不能运行时 patch

运行时改 `m.geom_contype[g] / m.geom_conaffinity[g]` **不影响碰撞**——MuJoCo 在 `compile()` 时把 pair-filter 缓存了。因此必须走 `mujoco.MjSpec` 路径，在编译前修改再编译。

#### 碰撞位掩码方案

`load_model_with_collision_filter(xml, arm)` 在编译期（`arm` 可为 `"left"`/`"right"`/`"both"`）：

- 手臂远端链路（`{arm}_shoulder_roll_link` 及以下，左手包含五指各段）：`contype=1, conaffinity=2`；
- `head_yaw_link / head_pitch_link`：`contype=2, conaffinity=1`；
- 其它 body 显式清零，避免相邻骨段永久接触产生噪声。

#### 乐器自动检测与碰撞 primitive 注册

与真机 ``--urdf`` 规则一致，按 **模型 XML 文件名的 stem 后缀** 推断（见
`hard_stop_calibration.detect_instrument_from_xml_path`）：``*_bass.xml``、``*_guitar.xml``、
``*_keyboard.xml`` 或裸机（无上述后缀）。不再根据 mesh 名或 worldbody 中的 body 名推断。

对于 guitar/bass（接触类），**删除**原始 mesh collider（MuJoCo 凸包远大于真实轮廓），然后注入对应的 box primitive：

| primitive | bare | guitar | bass |
|-----------|------|--------|------|
| `waist_torso_collider` | ✓ center(0.02, 0, 0.18) half(0.11, 0.115, 0.19) | ✓ 同左 | ✓ 同左 |
| `{inst}_body_collider` | — | ✓ center(0.145, -0.125, -0.06) half(0.055, 0.275, 0.13) | ✓ center(0.12, -0.17, -0.09) half(0.05, 0.15, 0.14) |
| `{inst}_neck_collider` | — | ✓ center(0.162, 0.285, 0.08) half(0.018, 0.135, 0.06) | ✓ center(0.22, 0.31, 0.15) half(0.015, 0.16, 0.05) |

各尺寸通过对比 `waist_yaw_link.STL`（纯躯干）与 `waist_yaw_link_{instrument}.STL`（躯干+乐器）的顶点差集在 Y/Z 切片下的分布估算。

设计取舍：
- **torso box** Y 半轴 0.115（略窄于真实 ±0.145），避免 shoulder_roll_link 在肩窝处贴着 box 壁产生微接触。
- **乐器琴身** box 覆盖身体正前方低段，只拦挡"手臂横跨身体穿过琴身"的路径。
- **乐器琴颈** box 很细，拦住"手臂直接穿过琴颈"的情况。
- **背带(strap)** 不加碰撞——柔性织物，与左臂 hanging 空间完全重合，加了就无法让手臂回到自然姿态。

#### 验证结果

四种模型 × 双臂（`--arm both`）标定结果：

| 模型 | shoulder_roll offset | 其余 6 关节 max offset | 双臂总耗时 |
|------|---------------------|----------------------|-----------|
| guitar | ±0.030 rad | < 0.002 rad | ~14 s |
| bass | ±0.030 rad | < 0.002 rad | ~15 s |
| bare | ±0.030 rad | < 0.002 rad | ~13 s |
| keyboard | ±0.030 rad | < 0.002 rad | ~13 s |

`shoulder_roll` 的 ±0.03 rad 偏差来自躯干 box 在 URDF 硬限位前几毫米拦住手臂（物理上合理——现实中手臂也会先碰到身体）。

#### keyboard（电子琴）的特殊处理

电子琴模型与 guitar/bass 不同：琴体以独立 body（含数十个琴键铰链关节和 box 几何体）放置在 worldbody 中，而非融入 `waist_yaw_link` 的 mesh。由于电子琴距机器人较远（约 0.5 m），手臂标定轨迹不会接触琴体，因此：

- **编译期可选剥离**：`load_model_with_collision_filter(strip_standalone_instruments=…)` 默认在**无头且未录像**时删除 `keyboard` 子树以提速；使用 `--visualize` 或 `--record` 时**不再剥离**，电子琴会出现在画面/视频里。无头又需要看到琴体时可加 `--keep-keyboard`；若要在可视化时仍剥离（省算力）用 `--strip-keyboard`。
- **求解器重置**：keyboard XML 含有为琴键接触优化的重型求解器设置（`implicitfast`, `cone=elliptic`, `tolerance=1e-9`, `iterations=150`），标定不需要，一律重置为 Euler + pyramidal 默认值。
- **标定姿态**：`keyboard` 在规划上归一为 guitar 高抬臂避障（`plan_instrument=guitar`），与实机电子琴在身前一致。

### 扩展新乐器

如需支持新的接触类乐器（如另一种弦乐器），步骤：

1. 准备 `waist_yaw_link_{name}.STL`、对应的 XML 和 URDF；
2. 用 `waist_yaw_link.STL`（纯躯干）做顶点差集分析，确定琴身/琴颈的 AABB；
3. 在 `mujoco_hard_stop_calibration.py` 的 `_INSTRUMENT_COLLIDERS` 字典中添加 `"{name}": [...]`；
4. 跑 48 项静态姿态检查（模型 × 2 臂 × 8 poses）确认零碰撞；
5. 跑完整标定确认所有 offset < 0.03 rad。

如需支持新的独立乐器（如 drum），步骤：

1. 准备对应的 XML（琴体作为 worldbody 下独立 body）和 URDF（与 bare 相同）；
2. 在 `_STANDALONE_INSTRUMENT_BODIES` 集合中添加 body 名称；
3. 确认乐器距机器人足够远，标定轨迹不经过其空间。

## 建议的实机接入步骤

1. 先只跑 `hard_stop_calibration.py --print-plan` 或 `ros2_upper_body_hardware.py --print-plan-only`，人工检查关节顺序、目标限位方向和 holding pose 是否符合机构常识。
2. 接入单关节低速顶靠，先在人机安全条件满足、急停可触达下验证一个关节。
3. 调整 `HardStopDetectorConfig` 或 `--current-thresholds` 中的速度、电流/力矩与位置窗口阈值。
4. 单臂验证通过后，再决定零偏的持久化方式（电驱工具或内部服务）。

## 当前局限

- 碰撞 primitive 基于 mesh 顶点 AABB 手动估算，非精确贴合；shoulder_roll 约 ±0.03 rad 的偏差即来自此（现实中需在实机标定补偿）。
- 背带(strap)不加碰撞——现实中为柔性织物；仿真里 viewer 看到手臂与背带可视 mesh 交叠是预期行为。
- 乐器感知的避障姿态让手臂在 pitch ≈ -2.5 rad（高举过头）的位置运动，真机需确认此范围安全。
- `shoulder_roll` 的 `preferred_stop_side` 选在 ±0.3491 rad 一侧；现场若只在宽端装了硬停，需显式覆盖并先做碰撞路径评估。
- ROS2 路径依赖现场 topic 与 `UpperJointData` 字段定义；与产线 msg 若不一致，需在 `ros2_upper_body_hardware.py` 中做少量对齐。
- 无自带急停与故障恢复，须在现场安全流程内使用。

若你现场 `UpperJointData` 的 `.msg` 字段与产线包不一致，在 `ros2_upper_body_hardware.py` 中改发布处即可。关节反馈 topic 在文档 2.5.3 已规定为 `/joint_states`，仅 remap 场景需改参。
