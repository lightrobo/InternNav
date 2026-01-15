#!/bin/bash
#===============================================================================
# RealSense SDK 一键安装脚本 - Jetson AGX Orin
# 
# 基于: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
# 支持: JetPack 5.0.2+ (L4T 35.1+)
#
# 用法:
#   ./install_realsense_sdk.sh          # 默认: RSUSB后端(推荐,~15分钟)
#   ./install_realsense_sdk.sh --native # Native V4L后端(含内核补丁,~30分钟)
#===============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 全局变量
WORK_DIR="/tmp/librealsense_build"
SYSTEM_PYTHON="/usr/bin/python3"

#===============================================================================
# 系统检测
#===============================================================================
check_system() {
    log_info "检测系统环境..."
    
    # 检查是否为Jetson设备
    if [ ! -f /etc/nv_tegra_release ]; then
        log_error "未检测到 NVIDIA Jetson 设备!"
        log_error "此脚本仅适用于 Jetson AGX Orin / Xavier 等设备"
        exit 1
    fi
    
    # 读取L4T版本
    L4T_VERSION=$(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo "unknown")
    log_info "L4T Release: $L4T_VERSION"
    
    # 检查架构
    ARCH=$(uname -m)
    if [ "$ARCH" != "aarch64" ]; then
        log_error "不支持的架构: $ARCH (需要 aarch64)"
        exit 1
    fi
    log_ok "架构: $ARCH"
    
    # 检查Ubuntu版本
    UBUNTU_VERSION=$(lsb_release -cs 2>/dev/null || echo "unknown")
    log_info "Ubuntu Codename: $UBUNTU_VERSION"
    
    # 检查系统Python
    if [ -f "$SYSTEM_PYTHON" ]; then
        PYTHON_VERSION=$($SYSTEM_PYTHON --version 2>&1)
        log_info "系统 Python: $PYTHON_VERSION"
    else
        log_error "未找到系统 Python: $SYSTEM_PYTHON"
        exit 1
    fi
    
    # 检查磁盘空间 (至少需要3GB用于源码构建)
    AVAILABLE_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    log_info "可用磁盘空间: ${AVAILABLE_SPACE}GB"
    
    if [ "$AVAILABLE_SPACE" -lt 3 ]; then
        log_error "磁盘空间不足3GB，无法从源码构建"
        exit 1
    fi
    
    log_ok "系统检测完成"
    echo ""
}

#===============================================================================
# 安装依赖
#===============================================================================
install_dependencies() {
    log_info "安装基础依赖..."
    
    sudo apt-get update
    sudo apt-get install -y \
        git \
        libssl-dev \
        libusb-1.0-0-dev \
        libudev-dev \
        pkg-config \
        libgtk-3-dev \
        cmake \
        build-essential \
        python3-dev \
        libglfw3-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev
    
    log_ok "依赖安装完成"
    echo ""
}

#===============================================================================
# 清理冲突的包
#===============================================================================
cleanup_old_installation() {
    log_info "清理旧的安装..."
    
    # 卸载可能冲突的pip包
    pip3 uninstall pyrealsense2 -y 2>/dev/null || true
    $SYSTEM_PYTHON -m pip uninstall pyrealsense2 -y 2>/dev/null || true
    
    # 卸载Debian包(如果存在且需要重建)
    if dpkg -l | grep -q librealsense2; then
        log_info "发现已安装的 Debian 包，正在卸载..."
        sudo apt-get remove -y librealsense2* 2>/dev/null || true
    fi
    
    log_ok "清理完成"
    echo ""
}

#===============================================================================
# 设置udev规则
#===============================================================================
setup_udev_rules() {
    log_info "设置 udev 规则..."
    
    # 使用官方脚本设置udev规则
    if [ -d "$WORK_DIR/librealsense" ]; then
        cd "$WORK_DIR/librealsense"
        sudo ./scripts/setup_udev_rules.sh
    else
        # 手动创建基本规则
        sudo bash -c 'cat > /etc/udev/rules.d/99-realsense-libusb.rules << EOF
# Intel RealSense D400 series
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666", GROUP="plugdev"
EOF'
        sudo udevadm control --reload-rules
        sudo udevadm trigger
    fi
    
    # 确保用户在plugdev组
    sudo usermod -aG plugdev $USER 2>/dev/null || true
    
    log_ok "udev 规则已设置"
    echo ""
}

#===============================================================================
# 从源码构建 - RSUSB 后端 (推荐,不需要内核补丁)
#===============================================================================
build_rsusb() {
    log_info "从源码构建 RealSense SDK (RSUSB 后端)..."
    log_info "此模式不需要内核补丁，兼容性更好"
    
    # 清理旧的构建目录
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    # 克隆仓库
    log_info "克隆 librealsense 仓库..."
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git
    cd librealsense
    
    # 设置udev规则
    setup_udev_rules
    
    # 构建SDK
    log_info "配置构建..."
    mkdir -p build && cd build
    
    # 检测是否有CUDA
    BUILD_CUDA="false"
    if command -v nvcc &> /dev/null; then
        log_info "检测到 CUDA，启用 CUDA 加速"
        BUILD_CUDA="true"
    fi
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=true \
        -DBUILD_GRAPHICAL_EXAMPLES=true \
        -DFORCE_RSUSB_BACKEND=ON \
        -DBUILD_WITH_CUDA=$BUILD_CUDA \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE=$SYSTEM_PYTHON
    
    # 使用 (nproc - 1) 个核心编译
    JOBS=$(($(nproc) - 1))
    [ "$JOBS" -lt 1 ] && JOBS=1
    
    log_info "开始编译 (使用 $JOBS 核心，约需10-15分钟)..."
    make -j$JOBS
    
    log_info "安装..."
    sudo make install
    
    # 更新库缓存
    sudo ldconfig
    
    # 安装Python绑定到系统Python
    log_info "安装 Python 绑定..."
    PYREALSENSE_SO=$(find . -name "pyrealsense2*.so" -type f | head -1)
    if [ -n "$PYREALSENSE_SO" ]; then
        PYTHON_SITE=$($SYSTEM_PYTHON -c "import site; print(site.getsitepackages()[0])")
        sudo cp "$PYREALSENSE_SO" "$PYTHON_SITE/"
        sudo cp pybackend2*.so "$PYTHON_SITE/" 2>/dev/null || true
        log_ok "Python 绑定已安装到: $PYTHON_SITE"
    fi
    
    log_ok "RSUSB 后端构建完成"
    echo ""
}

#===============================================================================
# 从源码构建 - Native V4L 后端 (含内核补丁)
#===============================================================================
build_native() {
    log_info "从源码构建 RealSense SDK (Native V4L 后端)..."
    log_warn "此模式需要应用内核补丁，约需30分钟"
    log_warn "请勿中断此过程!"
    
    # 清理旧的构建目录
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    # 克隆仓库
    log_info "克隆 librealsense 仓库..."
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git
    cd librealsense
    
    # 设置udev规则
    setup_udev_rules
    
    # 运行内核补丁 (针对Jetson L4T)
    log_info "应用内核补丁 (这可能需要20-30分钟)..."
    
    if [ -f "./scripts/patch-realsense-ubuntu-L4T.sh" ]; then
        sudo ./scripts/patch-realsense-ubuntu-L4T.sh
    else
        log_error "未找到 L4T 补丁脚本"
        exit 1
    fi
    
    # 构建SDK
    log_info "配置构建..."
    mkdir -p build && cd build
    
    # 检测是否有CUDA
    BUILD_CUDA="false"
    if command -v nvcc &> /dev/null; then
        log_info "检测到 CUDA，启用 CUDA 加速"
        BUILD_CUDA="true"
    fi
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=true \
        -DBUILD_GRAPHICAL_EXAMPLES=true \
        -DFORCE_RSUSB_BACKEND=OFF \
        -DBUILD_WITH_CUDA=$BUILD_CUDA \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE=$SYSTEM_PYTHON
    
    # 使用 (nproc - 1) 个核心编译
    JOBS=$(($(nproc) - 1))
    [ "$JOBS" -lt 1 ] && JOBS=1
    
    log_info "开始编译 (使用 $JOBS 核心)..."
    make -j$JOBS
    
    log_info "安装..."
    sudo make install
    
    # 更新库缓存
    sudo ldconfig
    
    # 安装Python绑定到系统Python
    log_info "安装 Python 绑定..."
    PYREALSENSE_SO=$(find . -name "pyrealsense2*.so" -type f | head -1)
    if [ -n "$PYREALSENSE_SO" ]; then
        PYTHON_SITE=$($SYSTEM_PYTHON -c "import site; print(site.getsitepackages()[0])")
        sudo cp "$PYREALSENSE_SO" "$PYTHON_SITE/"
        sudo cp pybackend2*.so "$PYTHON_SITE/" 2>/dev/null || true
        log_ok "Python 绑定已安装到: $PYTHON_SITE"
    fi
    
    log_ok "Native V4L 后端构建完成"
    echo ""
}

#===============================================================================
# 验证安装
#===============================================================================
verify_installation() {
    log_info "验证安装..."
    
    # 检查realsense-viewer
    if command -v realsense-viewer &> /dev/null; then
        log_ok "realsense-viewer 已安装"
    else
        log_warn "realsense-viewer 未找到"
    fi
    
    # 检查SDK版本
    if command -v rs-enumerate-devices &> /dev/null; then
        log_info "SDK 工具版本:"
        rs-enumerate-devices --version 2>/dev/null || true
    fi
    
    # 检查Python绑定
    log_info "检查 Python 绑定..."
    $SYSTEM_PYTHON -c "import pyrealsense2 as rs; print(f'  pyrealsense2 已加载')" 2>/dev/null && \
        log_ok "Python 绑定正常" || \
        log_warn "Python 绑定未就绪"
    
    # 检测设备
    log_info "检测 RealSense 设备..."
    $SYSTEM_PYTHON << 'EOF' 2>&1 || log_warn "设备枚举失败"
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print(f'  检测到 {len(devices)} 个设备')
for i, dev in enumerate(devices):
    try:
        name = dev.get_info(rs.camera_info.name)
        sn = dev.get_info(rs.camera_info.serial_number)
        print(f'  [{i}] {name} (SN: {sn})')
    except Exception as e:
        print(f'  [{i}] 设备存在但无法读取详情: {e}')
EOF
    
    echo ""
}

#===============================================================================
# 打印使用说明
#===============================================================================
print_usage() {
    echo ""
    log_ok "=============================================="
    log_ok "  RealSense SDK 安装完成!"
    log_ok "=============================================="
    echo ""
    echo "常用命令:"
    echo "  realsense-viewer          # 图形界面查看器"
    echo "  rs-enumerate-devices      # 列出设备"
    echo "  rs-depth                  # 深度流测试"
    echo ""
    echo "Python 使用 (请使用系统Python):"
    echo "  $SYSTEM_PYTHON your_script.py"
    echo ""
    echo "  示例代码:"
    echo "    import pyrealsense2 as rs"
    echo "    pipeline = rs.pipeline()"
    echo "    pipeline.start()"
    echo ""
    echo "如果设备无法识别，请尝试:"
    echo "  1. 重新插拔相机 (使用 USB 3.0 端口)"
    echo "  2. 运行: sudo udevadm control --reload-rules && sudo udevadm trigger"
    echo "  3. 重新登录 (使 plugdev 组生效)"
    echo ""
    
    if [ "$INSTALL_MODE" == "rsusb" ]; then
        log_info "当前使用 RSUSB 后端"
        log_info "如需 frame metadata 等高级功能，可运行:"
        log_info "  ./install_realsense_sdk.sh --native"
    fi
    
    echo ""
}

#===============================================================================
# 主函数
#===============================================================================
main() {
    echo ""
    echo "========================================"
    echo "  RealSense SDK 安装脚本"
    echo "  目标平台: Jetson AGX Orin"
    echo "========================================"
    echo ""
    
    # 解析参数
    INSTALL_MODE="rsusb"  # 默认使用RSUSB后端
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --native|-n)
                INSTALL_MODE="native"
                shift
                ;;
            --rsusb|-r)
                INSTALL_MODE="rsusb"
                shift
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "选项:"
                echo "  (无参数)      RSUSB 后端 (推荐,不需要内核补丁,~15分钟)"
                echo "  --rsusb, -r   同上"
                echo "  --native, -n  Native V4L 后端 (含内核补丁,~30分钟)"
                echo "  --help, -h    显示帮助"
                echo ""
                echo "推荐先试 RSUSB 模式，如果有问题再用 --native"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    done
    
    # 检查root权限
    if [ "$EUID" -eq 0 ]; then
        log_warn "请不要以 root 身份运行此脚本"
        log_warn "脚本会在需要时自动请求 sudo 权限"
        exit 1
    fi
    
    # 执行安装
    check_system
    install_dependencies
    cleanup_old_installation
    
    if [ "$INSTALL_MODE" == "native" ]; then
        log_info "安装模式: Native V4L 后端 (含内核补丁)"
        build_native
    else
        log_info "安装模式: RSUSB 后端 (推荐)"
        build_rsusb
    fi
    
    verify_installation
    print_usage
    
    log_ok "安装脚本执行完毕!"
}

# 执行主函数
main "$@"
