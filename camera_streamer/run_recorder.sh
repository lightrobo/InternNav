#!/bin/bash
#
# RealSense RGBD+IMU rosbag 录制启动脚本
#
# 用法:
#   ./run_recorder.sh                     # 默认录制
#   ./run_recorder.sh -d 60               # 录制60秒
#   ./run_recorder.sh -o /path/to/out.bag # 指定输出路径
#   ./run_recorder.sh --no-imu            # 仅RGBD
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认输出目录
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "$DATA_DIR"

# 默认输出文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUTPUT="${DATA_DIR}/realsense_${TIMESTAMP}.bag"

# 检查 ROS 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "[INFO] 未检测到 ROS 环境，尝试自动加载..."
    
    # 尝试常见的 ROS 安装路径
    if [ -f "/opt/ros/noetic/setup.bash" ]; then
        source /opt/ros/noetic/setup.bash
        echo "[INFO] 已加载 ROS Noetic"
    elif [ -f "/opt/ros/melodic/setup.bash" ]; then
        source /opt/ros/melodic/setup.bash
        echo "[INFO] 已加载 ROS Melodic"
    else
        echo "[WARN] 未找到 ROS 安装，部分功能可能不可用"
        echo "[WARN] 请手动 source ROS 环境: source /opt/ros/<distro>/setup.bash"
    fi
else
    echo "[INFO] 当前 ROS 环境: $ROS_DISTRO"
fi

# 检查 Python 依赖
echo "[INFO] 检查依赖..."
python3 -c "import pyrealsense2" 2>/dev/null || {
    echo "[ERROR] pyrealsense2 未安装"
    echo "请运行: pip install pyrealsense2"
    exit 1
}

python3 -c "import rosbag" 2>/dev/null || {
    echo "[ERROR] rosbag 未安装"
    echo "请运行: pip install rosbag rospkg"
    echo "或确保已 source ROS 环境"
    exit 1
}

echo "[INFO] 依赖检查通过"
echo ""

# 如果没有指定输出文件，使用默认值
if [[ ! "$*" =~ "-o" ]] && [[ ! "$*" =~ "--output" ]]; then
    echo "[INFO] 输出文件: $DEFAULT_OUTPUT"
    python3 rosbag_recorder.py -o "$DEFAULT_OUTPUT" "$@"
else
    python3 rosbag_recorder.py "$@"
fi
