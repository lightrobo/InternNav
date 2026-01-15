#!/usr/bin/env python3
"""检测RealSense设备状态 - 简化版"""
import pyrealsense2 as rs

print(f"pyrealsense2 loaded from: {rs.__file__}")

print("\n1. 检测设备数量...")
ctx = rs.context()
devices = ctx.query_devices()
print(f"   设备数量: {len(devices)}")

if len(devices) == 0:
    print("❌ 未检测到设备")
    exit(1)

print("\n2. 尝试直接启动pipeline (使用默认配置)...")
try:
    pipeline = rs.pipeline()
    # 不指定任何配置，让realsense自己选择
    profile = pipeline.start()
    print("✅ Pipeline启动成功!")
    
    # 获取设备信息
    device = profile.get_device()
    print(f"   设备: {device.get_info(rs.camera_info.name)}")
    
    # 获取一帧
    print("\n3. 获取帧...")
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    
    if depth:
        print(f"   ✅ Depth: {depth.get_width()}x{depth.get_height()}")
    if color:
        print(f"   ✅ Color: {color.get_width()}x{color.get_height()}")
    
    pipeline.stop()
    print("\n✅ 测试完成!")
    
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")
    
    print("\n尝试方案B: 使用config明确指定...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        # 尝试更保守的配置
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)
        profile = pipeline.start(config)
        print("✅ 方案B成功!")
        pipeline.stop()
    except Exception as e2:
        print(f"❌ 方案B也失败: {e2}")
        
        print("\n可能需要安装NVIDIA官方的librealsense:")
        print("  参考: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md")
