"""默认资源路径解析：优先走已安装包的 ``share/`` 目录，回落到仓库源码布局。

设计动机
--------
1. 以 ``ros2 run zero_offset_calibration ...`` 启动时，工作目录不再是源码根，
   老的 ``casbot_band_urdf/urdf/xxx.urdf`` 相对路径会失效；需要通过
   :func:`ament_index_python.get_package_share_directory` 解析到
   ``install/share/zero_offset_calibration/``。
2. 未安装（例如 ``python3 -m zero_offset_calibration.hard_stop_calibration``
   直接在源码内跑）时，再尝试源码布局：
   ``<pkg>/.. (= src/zero_offset_calibration)/casbot_band_urdf/<kind>/<name>``。
3. 上述两处都找不到时，保留原始的相对路径字符串，由 argparse 抛出
   ``FileNotFoundError`` 时用户可见清晰来源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_PACKAGE_NAME = "zero_offset_calibration"


def _share_root() -> Optional[Path]:
    """返回已安装的 ``share/zero_offset_calibration`` 目录；未安装返回 ``None``。"""
    try:
        from ament_index_python.packages import (  # type: ignore[import-not-found]
            PackageNotFoundError,
            get_package_share_directory,
        )
    except Exception:
        return None
    try:
        share = Path(get_package_share_directory(_PACKAGE_NAME))
    except PackageNotFoundError:
        return None
    except Exception:
        return None
    return share if share.is_dir() else None


def _source_root() -> Optional[Path]:
    """返回源码仓内的 ``src/zero_offset_calibration/`` 目录；不存在返回 ``None``。"""
    here = Path(__file__).resolve().parent.parent
    return here if (here / "casbot_band_urdf").is_dir() else None


def _resolve(kind: str, name: str) -> Path:
    """按 ``share/`` → 源码 → 相对路径 三挡查找 ``casbot_band_urdf/<kind>/<name>``。"""
    for root in (_share_root(), _source_root()):
        if root is None:
            continue
        candidate = root / "casbot_band_urdf" / kind / name
        if candidate.is_file():
            return candidate
    return Path("casbot_band_urdf") / kind / name


def default_urdf_path(name: str) -> Path:
    """默认 URDF 路径，供各 CLI 的 ``--urdf`` 作为 ``default=``。"""
    return _resolve("urdf", name)


def default_xml_path(name: str) -> Path:
    """默认 MuJoCo XML 路径，供 ``mujoco_hard_stop_calibration`` 的 ``--model-xml``。"""
    return _resolve("xml", name)
