#!/usr/bin/env python3
"""
RealSense D435i RGBD + IMU -> ROS1 Bag 录制工具

将 RealSense 的 RGB、Depth、Accel、Gyro 数据保存为 rosbag 格式
无需运行 ROS Master，直接使用 rosbag Python API

依赖：
    pip install pyrealsense2 rosbag rospkg
    # 如果没有 ROS 环境，需要额外安装：
    pip install sensor-msgs-py  # 或从 ROS 安装

用法：
    python rosbag_recorder.py --output ./data/test.bag
    python rosbag_recorder.py --output ./data/test.bag --no-imu  # 仅 RGBD
    python rosbag_recorder.py --output ./data/test.bag --duration 60  # 录制60秒

作者：InternNav Team
"""

import os
import sys
import time
import argparse
import signal
from datetime import datetime
from typing import Optional, Tuple
from threading import Event

import numpy as np
import pyrealsense2 as rs

# ROS 相关导入
try:
    import rosbag
    import rospy
    from sensor_msgs.msg import Image, Imu, CameraInfo
    from std_msgs.msg import Header
    from geometry_msgs.msg import Vector3
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[ERROR] ROS 依赖未安装！")
    print("请安装: pip install rosbag rospkg")
    print("或在 ROS 环境中运行此脚本")


class RosbagRecorder:
    """RealSense D435i RGBD+IMU 录制到 rosbag"""

    def __init__(
        self,
        output_path: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_imu: bool = True,
        align_depth: bool = True,
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_imu = enable_imu
        self.align_depth = align_depth

        # RealSense
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.depth_scale: float = 0.001

        # ROS bag
        self.bag: Optional[rosbag.Bag] = None

        # 统计
        self.frame_count = 0
        self.imu_count = 0
        self.start_time = 0.0
        
        # 停止信号
        self.stop_event = Event()

    def _detect_device(self) -> Tuple[bool, bool]:
        """
        检测设备并确定是否支持 IMU
        
        Returns:
            (has_device, has_imu): 是否有设备，是否有IMU
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            return False, False

        device = devices[0]
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"[REC] 检测到设备: {name} (SN: {serial})")

        # D435i/D455 有 IMU，D435 没有
        has_imu = 'D435i' in name or 'D455' in name or 'D405' in name
        if self.enable_imu and not has_imu:
            print(f"[REC] ⚠️  警告: {name} 不支持 IMU，将仅录制 RGBD")
            self.enable_imu = False

        return True, has_imu

    def open_camera(self) -> bool:
        """初始化 RealSense 相机"""
        print(f"[REC] 初始化 RealSense...")

        has_device, has_imu = self._detect_device()
        if not has_device:
            print("[REC] ❌ 未检测到 RealSense 设备!")
            return False

        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # 配置 RGB 和 Depth 流
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )

            # 配置 IMU 流（如果支持）
            if self.enable_imu:
                # 加速度计：最高 250Hz
                self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
                # 陀螺仪：最高 400Hz
                self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
                print("[REC] IMU 流已启用 (Accel@250Hz, Gyro@400Hz)")

            # 启动 pipeline
            profile = self.pipeline.start(self.config)

            # 获取深度比例
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"[REC] 深度比例: {self.depth_scale}")

            # 深度对齐
            if self.align_depth:
                self.align = rs.align(rs.stream.color)
                print("[REC] 深度对齐已启用")

            # 获取相机内参
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            
            self.color_intrinsics = color_profile.get_intrinsics()
            self.depth_intrinsics = depth_profile.get_intrinsics()

            print(f"[REC] RGB: {color_profile.width()}x{color_profile.height()} @ {color_profile.fps()}fps")
            print(f"[REC] Depth: {depth_profile.width()}x{depth_profile.height()} @ {depth_profile.fps()}fps")

            # 预热
            print("[REC] 预热中...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            print("[REC] 预热完成")

            return True

        except Exception as e:
            print(f"[REC] ❌ 初始化失败: {e}")
            return False

    def open_bag(self) -> bool:
        """打开 rosbag 文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            
            self.bag = rosbag.Bag(self.output_path, 'w')
            print(f"[REC] 创建 bag 文件: {self.output_path}")
            return True
        except Exception as e:
            print(f"[REC] ❌ 创建 bag 文件失败: {e}")
            return False

    def _create_header(self, timestamp: float, frame_id: str) -> Header:
        """创建 ROS Header"""
        header = Header()
        header.stamp = rospy.Time.from_sec(timestamp)
        header.frame_id = frame_id
        return header

    def _create_image_msg(
        self,
        data: np.ndarray,
        timestamp: float,
        frame_id: str,
        encoding: str,
    ) -> Image:
        """创建 sensor_msgs/Image 消息"""
        msg = Image()
        msg.header = self._create_header(timestamp, frame_id)
        msg.height = data.shape[0]
        msg.width = data.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = False
        
        if len(data.shape) == 3:
            msg.step = data.shape[1] * data.shape[2]
        else:
            msg.step = data.shape[1] * data.dtype.itemsize
            
        msg.data = data.tobytes()
        return msg

    def _create_camera_info(self, intrinsics, timestamp: float, frame_id: str) -> CameraInfo:
        """创建 sensor_msgs/CameraInfo 消息"""
        msg = CameraInfo()
        msg.header = self._create_header(timestamp, frame_id)
        msg.height = intrinsics.height
        msg.width = intrinsics.width
        msg.distortion_model = "plumb_bob"
        
        # 畸变系数 [k1, k2, p1, p2, k3]
        msg.D = list(intrinsics.coeffs)
        
        # 内参矩阵 K (3x3)
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        msg.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        
        # 整流矩阵 R (单位矩阵)
        msg.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        
        # 投影矩阵 P (3x4)
        msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        
        return msg

    def _create_imu_msg(
        self,
        accel_data: Optional[Tuple[float, float, float]],
        gyro_data: Optional[Tuple[float, float, float]],
        timestamp: float,
    ) -> Imu:
        """创建 sensor_msgs/Imu 消息"""
        msg = Imu()
        msg.header = self._create_header(timestamp, "imu_link")

        # 加速度 (m/s^2)
        if accel_data:
            msg.linear_acceleration.x = accel_data[0]
            msg.linear_acceleration.y = accel_data[1]
            msg.linear_acceleration.z = accel_data[2]

        # 角速度 (rad/s)
        if gyro_data:
            msg.angular_velocity.x = gyro_data[0]
            msg.angular_velocity.y = gyro_data[1]
            msg.angular_velocity.z = gyro_data[2]

        # 协方差（未知，设为 -1）
        msg.orientation_covariance[0] = -1
        msg.angular_velocity_covariance[0] = -1 if not gyro_data else 0
        msg.linear_acceleration_covariance[0] = -1 if not accel_data else 0

        return msg

    def record(self, duration: Optional[float] = None):
        """
        开始录制
        
        Args:
            duration: 录制时长（秒），None 表示持续录制直到手动停止
        """
        if not HAS_ROS:
            print("[REC] ❌ ROS 依赖不可用")
            return

        if not self.open_camera():
            return

        if not self.open_bag():
            return

        # 设置信号处理
        def signal_handler(sig, frame):
            print("\n[REC] 收到停止信号...")
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"[REC] 🔴 开始录制{'（时长: ' + str(duration) + '秒）' if duration else ''}...")
        print("[REC] 按 Ctrl+C 停止")

        self.start_time = time.time()
        last_status_time = self.start_time
        
        # 用于 IMU 同步的临时存储
        last_accel = None
        last_gyro = None

        try:
            while not self.stop_event.is_set():
                # 检查时长
                elapsed = time.time() - self.start_time
                if duration and elapsed >= duration:
                    print(f"\n[REC] 达到指定时长 {duration} 秒")
                    break

                # 获取帧（非阻塞方式获取所有可用帧）
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                timestamp = time.time()
                ros_time = rospy.Time.from_sec(timestamp)

                # 处理 IMU 数据（高频率）
                if self.enable_imu:
                    # 加速度计
                    accel_frame = frames.first_or_default(rs.stream.accel)
                    if accel_frame:
                        accel = accel_frame.as_motion_frame().get_motion_data()
                        last_accel = (accel.x, accel.y, accel.z)
                        
                        # 创建仅有加速度的 IMU 消息
                        imu_msg = self._create_imu_msg(last_accel, None, timestamp)
                        self.bag.write('/camera/accel/sample', imu_msg, ros_time)
                        self.imu_count += 1

                    # 陀螺仪
                    gyro_frame = frames.first_or_default(rs.stream.gyro)
                    if gyro_frame:
                        gyro = gyro_frame.as_motion_frame().get_motion_data()
                        last_gyro = (gyro.x, gyro.y, gyro.z)
                        
                        # 创建仅有角速度的 IMU 消息
                        imu_msg = self._create_imu_msg(None, last_gyro, timestamp)
                        self.bag.write('/camera/gyro/sample', imu_msg, ros_time)
                        self.imu_count += 1

                    # 合并的 IMU 消息（用最近的数据）
                    if last_accel and last_gyro:
                        imu_msg = self._create_imu_msg(last_accel, last_gyro, timestamp)
                        self.bag.write('/camera/imu', imu_msg, ros_time)

                # 处理图像数据
                if self.align:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if color_frame and depth_frame:
                    # RGB 图像
                    color_data = np.asanyarray(color_frame.get_data())
                    color_msg = self._create_image_msg(
                        color_data, timestamp, "camera_color_optical_frame", "rgb8"
                    )
                    self.bag.write('/camera/color/image_raw', color_msg, ros_time)

                    # 深度图像
                    depth_data = np.asanyarray(depth_frame.get_data())
                    depth_msg = self._create_image_msg(
                        depth_data, timestamp, "camera_depth_optical_frame", "16UC1"
                    )
                    self.bag.write('/camera/depth/image_raw', depth_msg, ros_time)

                    # 相机内参
                    color_info = self._create_camera_info(
                        self.color_intrinsics, timestamp, "camera_color_optical_frame"
                    )
                    depth_info = self._create_camera_info(
                        self.depth_intrinsics, timestamp, "camera_depth_optical_frame"
                    )
                    self.bag.write('/camera/color/camera_info', color_info, ros_time)
                    self.bag.write('/camera/depth/camera_info', depth_info, ros_time)

                    self.frame_count += 1

                # 状态输出
                current_time = time.time()
                if current_time - last_status_time >= 1.0:
                    elapsed = current_time - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    imu_rate = self.imu_count / elapsed if elapsed > 0 else 0
                    
                    status = f"\r[REC] 时长: {elapsed:.1f}s | "
                    status += f"帧数: {self.frame_count} ({fps:.1f} fps) | "
                    if self.enable_imu:
                        status += f"IMU: {self.imu_count} ({imu_rate:.0f} Hz)"
                    
                    print(status, end='', flush=True)
                    last_status_time = current_time

        except Exception as e:
            print(f"\n[REC] ❌ 录制错误: {e}")
        finally:
            self.close()

    def close(self):
        """关闭资源"""
        print("\n[REC] 正在保存...")
        
        if self.bag:
            self.bag.close()
            
            # 打印统计
            file_size = os.path.getsize(self.output_path)
            duration = time.time() - self.start_time
            
            print(f"[REC] ✅ 录制完成!")
            print(f"    文件: {self.output_path}")
            print(f"    大小: {file_size / 1024 / 1024:.2f} MB")
            print(f"    时长: {duration:.1f} 秒")
            print(f"    帧数: {self.frame_count}")
            if self.enable_imu:
                print(f"    IMU: {self.imu_count}")

        if self.pipeline:
            self.pipeline.stop()


def main():
    parser = argparse.ArgumentParser(
        description='RealSense D435i RGBD+IMU -> ROS bag 录制工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    # 基本录制
    python rosbag_recorder.py -o ./data/test.bag
    
    # 录制 60 秒
    python rosbag_recorder.py -o ./data/test.bag -d 60
    
    # 仅录制 RGBD（无 IMU）
    python rosbag_recorder.py -o ./data/test.bag --no-imu
    
    # 高分辨率录制
    python rosbag_recorder.py -o ./data/hd.bag -W 1280 -H 720 --fps 15
'''
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=f'./data/realsense_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bag',
        help='输出 bag 文件路径 (默认: ./data/realsense_YYYYMMDD_HHMMSS.bag)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=None,
        help='录制时长（秒），不指定则持续录制'
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        default=640,
        help='图像宽度 (默认: 640)'
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=480,
        help='图像高度 (默认: 480)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='帧率 (默认: 30)'
    )
    parser.add_argument(
        '--no-imu',
        action='store_true',
        help='禁用 IMU 录制（仅 RGBD）'
    )
    parser.add_argument(
        '--no-align',
        action='store_true',
        help='禁用深度对齐'
    )

    args = parser.parse_args()

    if not HAS_ROS:
        print("\n[ERROR] 无法导入 ROS 依赖，请确保：")
        print("  1. 已安装 ROS (推荐 ROS Noetic)")
        print("  2. 已 source ROS 环境: source /opt/ros/noetic/setup.bash")
        print("  3. 或安装独立包: pip install rosbag rospkg")
        sys.exit(1)

    recorder = RosbagRecorder(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_imu=not args.no_imu,
        align_depth=not args.no_align,
    )

    recorder.record(duration=args.duration)


if __name__ == '__main__':
    main()
