# 机器人手臂硬限位零偏校准说明

本仓库原先只有 `URDF/XML` 资产，没有控制或校准程序。现在新增了 `hard_stop_calibration.py`，用于两件事：

1. 从 `URDF` 提取左右臂 7 个关节的转轴、限位和默认校准顺序。
2. 提供一套“去硬限位 -> 检测触碰 -> 计算零偏 -> 回退卸力”的运行骨架，等待你接入底层电机/编码器接口。

## 适用关节链

左右臂各自按 7 自由度处理：

- `*_shoulder_pitch_joint`
- `*_shoulder_roll_joint`
- `*_shoulder_yaw_joint`
- `*_elbow_pitch_joint`
- `*_wrist_yaw_joint`
- `*_wrist_pitch_joint`
- `*_wrist_roll_joint`

默认基于 `casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf`。

## 校准思路

程序默认采用下面的零偏定义：

```text
joint_angle = encoder_sign * encoder_position + zero_offset
```

当关节碰到已知硬限位时，若该硬限位在模型中的参考角度为 `q_stop`，编码器读数为 `enc_stop`，则：

```text
zero_offset = q_stop - encoder_sign * enc_stop
```

其中 `encoder_sign` 需要你结合驱动方向定义填写，通常取 `+1` 或 `-1`。

## 硬限位检测逻辑

`HardStopCalibrator` 中的默认判据是：

- 持续给目标关节一个低速搜索速度；
- 最近一小段时间内编码器变化很小；
- 关节估计速度接近 0；
- 电机电流达到该关节阈值的一定比例；
- 满足以上条件后认为已经顶到机械硬限位。

这套判据是为了兼容大多数伺服驱动器的“低速顶靠”流程，但阈值必须根据实机调试。

## 默认姿态设计原则

默认姿态不是碰撞证明后的最优解，而是依据 `URDF` 结构生成的推荐值，原则如下：

- 肩关节优先把手臂外展，尽量远离躯干；
- 肘部默认保持一定弯曲，减少前臂扫到本体的概率；
- 腕部校准时，肩肘保持在稳定的中间姿态，只让末端关节单独搜索；
- 左右臂对称关节会选择不同的推荐硬限位方向，例如左右肩 roll、左右腕 roll。

因为仓库里没有碰撞模型求解或电机控制代码，这些姿态必须在实机上先做慢速验证。

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
  --urdf casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf \
  --arm left \
  --print-plan
```

查看右臂计划：

```bash
python3 hard_stop_calibration.py \
  --urdf casbot_band_urdf/urdf/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.urdf \
  --arm right \
  --print-plan
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
python3 ros2_upper_body_hardware.py --arm right --print-plan-only
python3 ros2_upper_body_hardware.py --arm right --persist
```

## MuJoCo 仿真标定

使用仓库内 `casbot_band_urdf/xml/CASBOT02_ENCOS_7dof_shell_20251015_P1L_guitar.xml`。腿部无执行器，仿真中在每一步将腿关节锁回初值。`move_to_pose` 在 PD 超时后会做一次运动学拉齐再阻尼，便于在仿真里尽快到达标定姿态。

依赖：`pip install mujoco`（本机已测 MuJoCo 3.3.x）。`read_sample` 使用**仿真时间戳**，与 `HardStopCalibrator` 配套；判停逻辑在 `hard_stop_calibration.py` 中对停留时长做了少量容差，避免固定步长导致永远达不到“满窗口时长”。

```bash
# 仅打印 URDF 计划
python3 mujoco_hard_stop_calibration.py --arm right --print-plan

# 跑通仿真并写出 zero_offsets_mujoco.yaml
python3 mujoco_hard_stop_calibration.py --arm right --out zero_offsets_mujoco.yaml

# 打开 MuJoCo 交互窗口，实时看机械臂各步顶靠与姿态（mujoco.viewer.launch_passive）
python3 mujoco_hard_stop_calibration.py --arm right --out zero_offsets_mujoco.yaml --visualize

# 降低 sync 频率加快速度：每 8 步仿真再刷新一帧
python3 mujoco_hard_stop_calibration.py --arm right --visualize --viewer-sync-every 8

# 注入常值编码器偏差（JSON），用于验证 offset ≈ -bias（sign=+1）
# echo '{"right_wrist_yaw_joint": 0.05}' > /tmp/bias.json
# python3 mujoco_hard_stop_calibration.py --arm right --encoder-bias /tmp/bias.json
```

无图形环境（如 CI）下不要用 `--visualize`；本机需可用 OpenGL/GLFW 的显示环境。

说明：仿真里“硬限位”即关节 `range` 约束，与真机机械硬停有差别；个别关节若与 URDF 限位细微不一致，可能出现略大的算得零偏。

## 建议的实机接入步骤

1. 先只跑 `hard_stop_calibration.py --print-plan` 或 `ros2_upper_body_hardware.py --print-plan-only`，人工检查关节顺序、目标限位方向和 holding pose 是否符合机构常识。
2. 接入单关节低速顶靠，先在人机安全条件满足、急停可触达下验证一个关节。
3. 调整 `HardStopDetectorConfig` 或 `--current-thresholds` 中的速度、电流/力矩与位置窗口阈值。
4. 单臂验证通过后，再决定零偏的持久化方式（电驱工具或内部服务）。

## 当前局限

- 目前仅根据 `URDF` 限位和左右对称关系给出推荐姿态，没有做碰撞检测。
- ROS2 路径依赖现场 topic 与 `UpperJointData` 字段定义；与产线 msg 若不一致，需在 `ros2_upper_body_hardware.py` 中做少量对齐。
- 无自带急停与故障恢复，须在现场安全流程内使用。

若你现场 `UpperJointData` 的 `.msg` 字段与产线包不一致，在 `ros2_upper_body_hardware.py` 中改发布处即可。关节反馈 topic 在文档 2.5.3 已规定为 `/joint_states`，仅 remap 场景需改参。
