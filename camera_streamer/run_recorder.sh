#!/bin/bash
#
# RealSense RGBD+IMU rosbag2 录制启动脚本
#
# 用法:
#   ./run_recorder.sh                     # 默认录制
#   ./run_recorder.sh -d 60               # 录制60秒
#   ./run_recorder.sh -o /path/to/output  # 指定输出路径 (目录，不需要 .bag 后缀)
#   ./run_recorder.sh --no-imu            # 仅RGBD
#
# 回放:
#   ros2 bag play ./data/realsense_xxx
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 强制使用系统 Python (ROS2 依赖系统 Python 3.10)
PYTHON="/usr/bin/python3"

# 默认输出目录
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "$DATA_DIR"

# 默认输出路径 (ROS2 bag 是目录，不需要 .bag 后缀)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUTPUT="${DATA_DIR}/realsense_${TIMESTAMP}"

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "[INFO] 未检测到 ROS 环境，尝试自动加载..."
    
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "[INFO] 已加载 ROS2 Humble"
    elif [ -f "/opt/ros/iron/setup.bash" ]; then
        source /opt/ros/iron/setup.bash
        echo "[INFO] 已加载 ROS2 Iron"
    else
        echo "[ERROR] 未找到 ROS2 安装"
        echo "请先安装 ROS2 或手动 source: source /opt/ros/humble/setup.bash"
        exit 1
    fi
else
    echo "[INFO] 当前 ROS 环境: $ROS_DISTRO"
fi

# 检查 Python 依赖
echo "[INFO] 检查依赖..."

$PYTHON -c "import pyrealsense2" 2>/dev/null || {
    echo "[ERROR] pyrealsense2 未安装"
    echo "请运行: ./install_realsense_sdk.sh"
    exit 1
}

$PYTHON -c "import rosbag2_py" 2>/dev/null || {
    echo "[ERROR] rosbag2_py 未安装"
    echo "请确保已 source ROS2 环境: source /opt/ros/humble/setup.bash"
    exit 1
}

echo "[INFO] 依赖检查通过"
echo ""

# 如果没有指定输出文件，使用默认值
if [[ ! "$*" =~ "-o" ]] && [[ ! "$*" =~ "--output" ]]; then
    echo "[INFO] 输出目录: $DEFAULT_OUTPUT"
    $PYTHON rosbag_recorder.py -o "$DEFAULT_OUTPUT" "$@"
else
    $PYTHON rosbag_recorder.py "$@"
fi
