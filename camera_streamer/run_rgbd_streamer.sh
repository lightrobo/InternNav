#!/bin/bash
# AGX Orin RealSense D435 RGBD采集端启动脚本
# 在 AGX Orin 上运行，通过 gRPC 发送 RGB+Depth 到云端推理

# ============ 配置区域 ============
SERVER_ADDR="10.8.200.42:50100"   # 推理服务器地址（需要修改为实际IP）
WIDTH=640                       # 图像宽度
HEIGHT=480                      # 图像高度
FPS=15                          # 目标帧率
JPEG_QUALITY=80                 # JPEG压缩质量 (1-100)
INSTRUCTION="Follow the person" # 文本指令
HEADLESS=true                   # 是否无头模式
HTTP_STREAM=true                # 是否启用HTTP视频流
HTTP_PORT=8080                  # HTTP流端口
NO_INFER=false                  # 预览模式（不连接服务器）
ALIGN_DEPTH=true                # 是否对齐深度图到RGB
NO_CONFIG=false                 # 是否禁用配置文件
CONFIG_PATH=""                  # 自定义配置文件路径（为空则使用默认）
# ================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config/config1.json"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-display)
            HEADLESS=true
            shift
            ;;
        --display)
            HEADLESS=false
            shift
            ;;
        --http-stream)
            HTTP_STREAM=true
            shift
            ;;
        --no-http-stream)
            HTTP_STREAM=false
            shift
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --server)
            SERVER_ADDR="$2"
            shift 2
            ;;
        --instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        --preview|--no-infer)
            NO_INFER=true
            shift
            ;;
        --no-align)
            ALIGN_DEPTH=false
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --no-config)
            NO_CONFIG=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 确定使用的配置文件
if [ "${NO_CONFIG}" = false ]; then
    if [ -z "${CONFIG_PATH}" ]; then
        CONFIG_PATH="${DEFAULT_CONFIG}"
    fi
fi

echo "=========================================="
echo "  RealSense D435 RGBD Streamer"
echo "=========================================="
echo ""
echo "配置:"
if [ "${NO_INFER}" = false ]; then
    echo "  服务器地址:   ${SERVER_ADDR}"
else
    echo "  服务器地址:   (预览模式，不连接)"
fi
if [ "${NO_CONFIG}" = false ]; then
    echo "  配置文件:     ${CONFIG_PATH}"
else
    echo "  配置文件:     (未启用)"
fi
echo "  JPEG质量:     ${JPEG_QUALITY}"
echo "  指令:         ${INSTRUCTION}"
echo "  无头模式:     ${HEADLESS}"
echo "  HTTP视频流:   ${HTTP_STREAM}"
echo "  深度对齐:     ${ALIGN_DEPTH}"
if [ "${HTTP_STREAM}" = true ]; then
    echo "  HTTP端口:     ${HTTP_PORT}"
fi
echo ""

# 构建命令
CMD="python3 ${SCRIPT_DIR}/rgbd_streamer.py"
CMD="${CMD} --server ${SERVER_ADDR}"
CMD="${CMD} --quality ${JPEG_QUALITY}"
CMD="${CMD} --instruction \"${INSTRUCTION}\""

if [ "${HEADLESS}" = true ]; then
    CMD="${CMD} --no-display"
fi

if [ "${HTTP_STREAM}" = true ]; then
    CMD="${CMD} --http-stream --http-port ${HTTP_PORT}"
fi

if [ "${NO_INFER}" = true ]; then
    CMD="${CMD} --no-infer"
fi

if [ "${ALIGN_DEPTH}" = false ]; then
    CMD="${CMD} --no-align"
fi

if [ "${NO_CONFIG}" = true ]; then
    CMD="${CMD} --no-config"
elif [ -n "${CONFIG_PATH}" ]; then
    CMD="${CMD} --config \"${CONFIG_PATH}\""
fi

echo "执行命令:"
echo "  ${CMD}"
echo ""
if [ "${HTTP_STREAM}" = true ]; then
    # 获取本机IP
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    echo "📺 HTTP视频流地址: http://${LOCAL_IP}:${HTTP_PORT}"
    echo ""
fi
if [ "${HEADLESS}" = true ]; then
    echo "按 Ctrl+C 退出"
else
    echo "按 'q' 退出"
fi
echo ""

# 执行
eval ${CMD}
