"""ament_python setup for :mod:`zero_offset_calibration`.

打包策略
--------
- Python 源码通过 ``packages=[package_name]`` 安装到 ``lib/python*/site-packages``。
- 运行时/规划期需要的 URDF、MuJoCo XML、STL 网格、以及示例 YAML 通过 ``data_files``
  安装到 ``share/zero_offset_calibration/`` 下，保持相对目录结构，
  方便运行时用 :func:`ament_index_python.get_package_share_directory` 解析。
- 三个 CLI 入口在 ``lib/zero_offset_calibration/`` 下生成 console script，
  ``ros2 run zero_offset_calibration <name>`` 可直接调用。
"""

from pathlib import Path

from setuptools import setup

package_name = "zero_offset_calibration"


def _collect(dir_rel: str) -> list[tuple[str, list[str]]]:
    """按 ``dir_rel`` 打平收集其下所有文件到 ``share/<pkg>/<相对路径>/``。"""
    root = Path(dir_rel)
    if not root.is_dir():
        return []
    entries: dict[str, list[str]] = {}
    for path in root.glob("**/*"):
        if not path.is_file():
            continue
        install_dir = f"share/{package_name}/{path.parent.as_posix()}"
        entries.setdefault(install_dir, []).append(path.as_posix())
    return sorted(entries.items())


data_files: list[tuple[str, list[str]]] = [
    (
        "share/ament_index/resource_index/packages",
        [f"resource/{package_name}"],
    ),
    (f"share/{package_name}", ["package.xml"]),
]

for sub in ("casbot_band_urdf", "config", "launch"):
    data_files.extend(_collect(sub))

readme = Path("README_calibration.md")
if readme.is_file():
    data_files.append((f"share/{package_name}", [readme.as_posix()]))


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CASBOT Team",
    maintainer_email="casbot@todo.todo",
    description=(
        "CASBOT upper-body hard-stop zero-offset calibration: planner, runtime, "
        "MuJoCo simulator adapter, and ROS 2 real-robot adapter."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 仅生成计划（无 ROS / MuJoCo 依赖）
            f"hard_stop_calibration = {package_name}.hard_stop_calibration:main",
            # MuJoCo 仿真标定入口（需 mujoco / numpy，通常只在 x86 开发机使用）
            f"mujoco_hard_stop_calibration = {package_name}.mujoco_hard_stop_calibration:main",
            # ROS 2 真机校准入口：/motion/upper_body_debug + UpperJointData
            f"ros2_upper_body_hardware = {package_name}.ros2_upper_body_hardware:main",
            # Web 控制台：浏览器操作标定流程
            f"calibration_web_ui = {package_name}.web_ui:main",
            # 动作数据播放器
            f"action_player = {package_name}.action_player:main",
        ],
    },
)
