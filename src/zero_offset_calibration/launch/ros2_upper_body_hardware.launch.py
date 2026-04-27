"""启动 CASBOT 上身硬限位零偏校准节点 (真机 ROS 2 版本)。

用法示例::

    ros2 launch zero_offset_calibration ros2_upper_body_hardware.launch.py arm:=right
    ros2 launch zero_offset_calibration ros2_upper_body_hardware.launch.py \
        arm:=left persist:=true skip_on_timeout:=true search_timeout:=45.0

该 launch 将 CLI 的主要参数暴露为 LaunchArgument，通过 OpaqueFunction 动态
装配成 ``ros2 run zero_offset_calibration ros2_upper_body_hardware ...``
的可执行参数。URDF 与 YAML 默认从 share/ 解析，可按需覆盖。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    def val(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    cli = [
        "--arm", val("arm"),
        "--instrument", val("instrument"),
        "--pose-speed-scale", val("pose_speed_scale"),
        "--joint-states", val("joint_states"),
        "--ros-joint-naming", val("ros_joint_naming"),
        "--move-tolerance", val("move_tolerance"),
        "--move-timeout", val("move_timeout"),
        "--move-stuck-window", val("move_stuck_window"),
        "--move-stuck-epsilon", val("move_stuck_epsilon"),
        "--search-mode", val("search_mode"),
        "--torque-nm", val("torque_nm"),
        "--damping", val("damping"),
        "--torque2vel", val("torque2vel"),
        "--torque-search-vmax", val("torque_search_vmax"),
        "--velocity-epsilon", val("velocity_epsilon"),
        "--position-epsilon", val("position_epsilon"),
        "--stall-time", val("stall_time"),
        "--search-timeout", val("search_timeout"),
        "--min-current-ratio", val("min_current_ratio"),
        "--backoff-seconds", val("backoff_seconds"),
        "--min-search-travel", val("min_search_travel"),
        "--max-expected-offset", val("max_expected_offset"),
        "--effort-baseline-seconds", val("effort_baseline_seconds"),
        "--effort-rise-nm", val("effort_rise_nm"),
        "--stuck-abort-seconds", val("stuck_abort_seconds"),
        "--default-current-threshold", val("default_current_threshold"),
        "--offsets-out", val("offsets_out"),
    ]
    if val("urdf"):
        cli += ["--urdf", val("urdf")]
    if val("encoder_signs"):
        cli += ["--encoder-signs", val("encoder_signs")]
    if val("current_thresholds"):
        cli += ["--current-thresholds", val("current_thresholds")]
    if val("persist").lower() in {"true", "1", "yes"}:
        cli += ["--persist"]
    if val("skip_on_timeout").lower() in {"true", "1", "yes"}:
        cli += ["--skip-on-timeout"]
    if val("skip_setup_waypoints").lower() in {"true", "1", "yes"}:
        cli += ["--skip-setup-waypoints"]
    if val("skip_reset_waypoints").lower() in {"true", "1", "yes"}:
        cli += ["--skip-reset-waypoints"]

    return [
        Node(
            package="zero_offset_calibration",
            executable="ros2_upper_body_hardware",
            name="casbot_zero_offset_calib",
            output="screen",
            arguments=cli,
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("arm", default_value="right", description="left/right"),
        DeclareLaunchArgument(
            "instrument",
            default_value="bass",
            description=(
                "乐器配置 (auto/none/bass/guitar/keyboard)：同时决定默认 URDF 和避障姿态。"
                "bass(默认)=_bass.urdf；guitar=_guitar.urdf；auto=按 --urdf 文件名识别；"
                "none=裸机；keyboard=guitar 高抬臂轨迹 + bass URDF。"
                "必须与真实穿戴一致，否则手臂会与琴体碰撞。"
            ),
        ),
        DeclareLaunchArgument(
            "pose_speed_scale",
            default_value="0.15",
            description="预备/收尾 waypoint 的 vel_scale，默认 0.15",
        ),
        DeclareLaunchArgument(
            "skip_setup_waypoints",
            default_value="false",
            description="跳过 rest→WP1→WP2→neutral 预备轨迹（不推荐）",
        ),
        DeclareLaunchArgument(
            "skip_reset_waypoints",
            default_value="false",
            description="跳过 neutral→rev-WP2→rest 收尾轨迹（不推荐）",
        ),
        DeclareLaunchArgument("urdf", default_value=""),
        DeclareLaunchArgument("joint_states", default_value="/joint_states"),
        DeclareLaunchArgument("ros_joint_naming", default_value="no_joint_suffix"),
        DeclareLaunchArgument("move_tolerance", default_value="0.05"),
        DeclareLaunchArgument("move_timeout", default_value="15.0"),
        DeclareLaunchArgument("move_stuck_window", default_value="3.0"),
        DeclareLaunchArgument("move_stuck_epsilon", default_value="0.010"),
        DeclareLaunchArgument("search_mode", default_value="torque_damping"),
        DeclareLaunchArgument("torque_nm", default_value="8.0"),
        DeclareLaunchArgument("damping", default_value="3.0"),
        DeclareLaunchArgument("torque2vel", default_value="0.012"),
        DeclareLaunchArgument("torque_search_vmax", default_value="0.12"),
        DeclareLaunchArgument("velocity_epsilon", default_value="0.08"),
        DeclareLaunchArgument("position_epsilon", default_value="0.012"),
        DeclareLaunchArgument("stall_time", default_value="0.50"),
        DeclareLaunchArgument("search_timeout", default_value="20.0"),
        DeclareLaunchArgument("min_current_ratio", default_value="0.10"),
        DeclareLaunchArgument("backoff_seconds", default_value="0.20"),
        DeclareLaunchArgument("min_search_travel", default_value="0.08"),
        DeclareLaunchArgument("max_expected_offset", default_value="0.60"),
        DeclareLaunchArgument("effort_baseline_seconds", default_value="0.30"),
        DeclareLaunchArgument("effort_rise_nm", default_value="0.15"),
        DeclareLaunchArgument("stuck_abort_seconds", default_value="4.0"),
        DeclareLaunchArgument("default_current_threshold", default_value="5.0"),
        DeclareLaunchArgument("encoder_signs", default_value=""),
        DeclareLaunchArgument("current_thresholds", default_value=""),
        DeclareLaunchArgument("offsets_out", default_value="src/config/joint_pos_offset.yaml"),
        DeclareLaunchArgument("persist", default_value="false"),
        DeclareLaunchArgument("skip_on_timeout", default_value="false"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
