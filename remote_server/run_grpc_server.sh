#!/bin/bash
# gRPC 推理服务器启动脚本
# 适配 camera_streamer 的 gRPC 客户端

set -e

# 默认参数
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50052}"
MODEL_DIR="${HF_MODEL_DIR:-}"
DEVICE="${DEVICE:-cuda}"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}gRPC 推理服务器${NC}"
echo -e "${CYAN}(适配 camera_streamer gRPC 客户端)${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src/diffusion-policy:$PYTHONPATH"

# 打印配置
echo -e "${YELLOW}配置:${NC}"
echo "  HOST: $HOST"
echo "  PORT: $PORT"
echo "  MODEL_DIR: ${MODEL_DIR:-'未设置 (将在首次推理时加载)'}"
echo "  DEVICE: $DEVICE"
echo ""

# 如果设置了 MODEL_DIR，导出环境变量
if [ -n "$MODEL_DIR" ]; then
    export HF_MODEL_DIR="$MODEL_DIR"
fi

# 构建启动命令
CMD="python remote_server/grpc_server.py --host $HOST --port $PORT --device $DEVICE"

if [ -n "$MODEL_DIR" ]; then
    CMD="$CMD --model-dir $MODEL_DIR"
fi

# 启动服务器
echo -e "${GREEN}启动 gRPC 服务器...${NC}"
echo ""

exec $CMD "$@"
