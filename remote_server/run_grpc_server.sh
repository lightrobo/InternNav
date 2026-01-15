#!/bin/bash
# gRPC 推理服务器启动脚本
# 使用 InternVLAN1AsyncAgent (实机部署版本)
# 适配 camera_streamer 的 gRPC 客户端

set -e

# 默认参数
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50052}"
MODEL_PATH="${MODEL_PATH:-${HF_MODEL_DIR:-}}"
DEVICE="${DEVICE:-cuda}"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}gRPC 推理服务器${NC}"
echo -e "${CYAN}InternVLAN1AsyncAgent (实机部署版)${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src/diffusion-policy:$PYTHONPATH"

# 检查模型路径
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 未设置模型路径!${NC}"
    echo ""
    echo "请使用以下方式之一设置模型路径:"
    echo "  1. MODEL_PATH=/path/to/model ./run_grpc_server.sh"
    echo "  2. export HF_MODEL_DIR=/path/to/model"
    echo "  3. ./run_grpc_server.sh --model-path /path/to/model"
    echo ""
    echo "模型下载地址:"
    echo "  https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN"
    echo "  https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP"
    exit 1
fi

# 打印配置
echo -e "${YELLOW}配置:${NC}"
echo "  HOST:       $HOST"
echo "  PORT:       $PORT"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  DEVICE:     $DEVICE"
echo ""

# 导出环境变量
export HF_MODEL_DIR="$MODEL_PATH"

# 构建启动命令
CMD="python remote_server/grpc_server.py --host $HOST --port $PORT --device $DEVICE --model-path $MODEL_PATH"

# 启动服务器
echo -e "${GREEN}启动 gRPC 服务器...${NC}"
echo ""

exec $CMD "$@"
