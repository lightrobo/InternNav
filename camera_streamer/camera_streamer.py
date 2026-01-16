#!/usr/bin/env python3
"""
AGX Orin 摄像头采集端 - gRPC Client
采集USB摄像头画面，通过gRPC发送到云端推理
支持 HTTP 视频流用于远程查看

服务器返回的是 waypoints（累积位移），不是速度！
- x: 前进方向位移 (m)，相对起点
- y: 左右方向位移 (m)，相对起点
- theta: 朝向角 (rad)，相对起点
"""

import cv2
import grpc
import time
import argparse
import numpy as np
from typing import Optional, Tuple
from threading import Thread, Lock
import math

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc

# ROS2 导入
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[Client] 警告: ROS2 未安装，运动控制功能不可用")
    print("[Client] 安装: sudo apt install ros-humble-rclpy ros-humble-std-msgs")


class MotionPublisher(Node):
    """ROS2 节点：发布速度命令到 /motion_ref"""
    
    def __init__(self):
        super().__init__('intern_nav_motion_publisher')
        self._publisher = self.create_publisher(Float32MultiArray, '/motion_ref', 1)
        self._vel_cmd_lock = Lock()
        self._vel_cmd = [0.0, 0.0, 0.0]  # [linear_x, linear_y, angular_z]
        self._publish_rate = 50.0  # Hz
        self._timer = self.create_timer(1.0 / self._publish_rate, self._timer_callback)
        self.get_logger().info(f'MotionPublisher 已启动，发布频率: {self._publish_rate}Hz')
    
    def update_velocity(self, linear_x: float, linear_y: float, angular_z: float):
        """更新速度命令"""
        with self._vel_cmd_lock:
            self._vel_cmd = [linear_x, linear_y, angular_z]
    
    def _timer_callback(self):
        """定时发布速度命令"""
        with self._vel_cmd_lock:
            vel_cmd = self._vel_cmd.copy()
        
        msg = Float32MultiArray()
        msg.data = vel_cmd
        self._publisher.publish(msg)


class CameraStreamer:
    """摄像头采集 + gRPC 客户端 + HTTP 视频流"""
    
    def __init__(
        self,
        server_addr: str = "localhost:50051",
        camera_idx: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 10,
        jpeg_quality: int = 80,
        http_port: int = 8080,
        enable_ros2: bool = True,
        vel_time_scale: float = 0.1
    ):
        self.server_addr = server_addr
        self.camera_idx = camera_idx
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.http_port = http_port
        self.enable_ros2 = enable_ros2 and ROS2_AVAILABLE
        self.vel_time_scale = vel_time_scale  # waypoint位移到速度的转换时间尺度
        
        self.channel = None
        self.stub = None
        self.cap = None
        self.frame_id = 0
        
        # HTTP 流相关
        self._current_frame = None
        self._frame_lock = Lock()
        self._http_server = None
        
        # ROS2 相关
        self._ros2_node = None
        self._ros2_thread = None
        self._rclpy_initialized = False
        
    def connect(self) -> bool:
        """连接gRPC服务器"""
        print(f"[Client] 连接服务器: {self.server_addr}")
        
        self.channel = grpc.insecure_channel(
            self.server_addr,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
        )
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        
        # 健康检查
        try:
            response = self.stub.HealthCheck(inference_pb2.Empty())
            print(f"[Client] 服务器状态: healthy={response.healthy}, device={response.device}")
            return response.healthy
        except grpc.RpcError as e:
            print(f"[Client] 连接失败: {e}")
            return False
    
    def open_camera(self) -> bool:
        """打开摄像头"""
        print(f"[Client] 打开摄像头: {self.camera_idx}")
        
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            print(f"[Client] 无法打开摄像头 {self.camera_idx}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[Client] 摄像头参数: {actual_w}x{actual_h} @ {actual_fps}fps")
        return True
    
    def encode_frame(self, frame: np.ndarray) -> bytes:
        """JPEG编码"""
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buffer.tobytes()
    
    def infer(self, frame: np.ndarray, instruction: str) -> Tuple[Optional[np.ndarray], float]:
        """发送帧到云端推理，返回 waypoints（累积位移序列）"""
        self.frame_id += 1
        
        # 编码
        image_data = self.encode_frame(frame)
        
        # 发送请求
        request = inference_pb2.InferRequest(
            image_data=image_data,
            instruction=instruction,
            frame_id=self.frame_id,
            timestamp_ms=int(time.time() * 1000)
        )
        
        try:
            response = self.stub.Infer(request)
            
            if response.success:
                # 调试：打印收到的原始数据
                print(f"[Client DEBUG] 收到: n_waypoints={response.n_waypoints}, waypoints长度={len(response.waypoints)}, 前9个值={list(response.waypoints[:9]) if len(response.waypoints) > 0 else []}")
                
                # 服务器返回的是累积位移: [x, y, theta] * n_waypoints
                if response.n_waypoints > 0 and len(response.waypoints) >= response.n_waypoints * 3:
                    waypoints = np.array(response.waypoints).reshape(response.n_waypoints, 3)
                else:
                    print(f"[Client] 警告: waypoints数据异常 - n_waypoints={response.n_waypoints}, 实际长度={len(response.waypoints)}")
                    return None, 0
                return waypoints, response.inference_time_ms
            else:
                print(f"[Client] 推理错误: {response.error}")
                return None, 0
        except grpc.RpcError as e:
            print(f"[Client] gRPC错误: {e}")
            return None, 0
    
    def draw_trajectory(self, frame: np.ndarray, waypoints: np.ndarray, scale: float = 120.0) -> np.ndarray:
        """
        在画面上绘制轨迹
        
        参数:
            frame: 原始图像
            waypoints: 累积位移序列 (N, 3) - [x, y, theta]
                       x: 前进方向位移 (m)
                       y: 左侧方向位移 (m)
                       theta: 朝向角 (rad)
            scale: 像素/米 的缩放比例 (默认120，与项目其他地方一致)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        # 与项目其他地方一致的投影参数
        cx = w // 2              # 图像宽度中心
        cy = int(h * 0.86)       # 图像高度的 86% 处作为机器人位置
        
        arrow_len = 20  # 方向箭头长度
        
        # 绘制轨迹点
        points = [(cx, cy)]  # 起点
        for i, (x, y, theta) in enumerate(waypoints):
            # 转换到屏幕坐标
            # 世界坐标: x=前（向上），y=左（向左）
            # 屏幕坐标: px向右增加，py向下增加
            px = int(cx - y * scale)  # y左 → px左
            py = int(cy - x * scale)  # x前 → py上
            points.append((px, py))
            
            # 画点
            # 颜色渐变：绿色（近）→ 红色（远）
            progress = i / max(len(waypoints) - 1, 1)
            color = (
                int(255 * progress),      # R: 远处变红
                int(255 * (1 - progress)), # G: 近处绿
                0
            )
            cv2.circle(vis, (px, py), 5, color, -1)
            cv2.circle(vis, (px, py), 7, (255, 255, 255), 1)
            
            # 画方向箭头（显示 theta）
            arrow_dx = int(-arrow_len * math.sin(theta))
            arrow_dy = int(-arrow_len * math.cos(theta))
            arrow_end = (px + arrow_dx, py + arrow_dy)
            
            # 箭头颜色：根据转向角度
            turn_intensity = min(abs(theta) / 0.5, 1.0)  # 0.5 rad ≈ 30° 作为最大
            arrow_color = (
                int(255 * turn_intensity),
                int(255 * (1 - turn_intensity * 0.5)),
                0
            )
            cv2.arrowedLine(vis, (px, py), arrow_end, arrow_color, 2, tipLength=0.4)
        
        # 连线（轨迹路径）
        for i in range(len(points) - 1):
            # 渐变线条颜色
            progress = i / max(len(points) - 2, 1)
            line_color = (
                int(100 * progress),
                int(200 * (1 - progress * 0.5)),
                0
            )
            cv2.line(vis, points[i], points[i + 1], line_color, 2)
        
        # 画机器人位置（底部中心 = 起点）
        cv2.circle(vis, (cx, cy), 12, (255, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 14, (255, 255, 255), 2)
        # 机器人朝向箭头（向上 = 初始朝向）
        cv2.arrowedLine(vis, (cx, cy), (cx, cy - 35), (255, 100, 100), 3, tipLength=0.3)
        cv2.putText(vis, "Robot", (cx - 25, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示第一个 waypoint 信息
        if len(waypoints) > 0:
            x, y, theta = waypoints[0]
            wp_info = f"WP0: x={x:.2f}m y={y:.2f}m th={math.degrees(theta):.1f}deg"
            cv2.putText(vis, wp_info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return vis
    
    def _update_http_frame(self, frame: np.ndarray):
        """更新 HTTP 流的当前帧"""
        with self._frame_lock:
            self._current_frame = frame.copy()
    
    def _generate_frames(self):
        """生成 MJPEG 帧流"""
        while True:
            with self._frame_lock:
                if self._current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self._current_frame.copy()
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30fps
    
    def _start_http_server(self):
        """启动 HTTP 视频流服务器"""
        try:
            from flask import Flask, Response
        except ImportError:
            print("[Client] 警告: Flask 未安装，HTTP 流功能不可用")
            print("[Client] 安装: pip install flask")
            return
        
        app = Flask(__name__)
        streamer = self
        
        @app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>OpenTrackVLA Live Stream</title>
                <style>
                    body { 
                        background: #1a1a2e; 
                        color: #eee; 
                        font-family: monospace;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        padding: 20px;
                    }
                    h1 { color: #00ff88; }
                    img { 
                        border: 2px solid #00ff88; 
                        border-radius: 8px;
                        max-width: 100%;
                    }
                    .info { 
                        margin-top: 10px; 
                        color: #888; 
                    }
                    .legend {
                        margin-top: 15px;
                        padding: 10px;
                        background: #2a2a4e;
                        border-radius: 5px;
                        font-size: 12px;
                    }
                </style>
            </head>
            <body>
                <h1>🤖 OpenTrackVLA Live Stream</h1>
                <img src="/video_feed" alt="Video Stream">
                <p class="info">实时推理可视化 | 累积位移轨迹</p>
                <div class="legend">
                    <b>图例:</b><br>
                    🔴 机器人位置（起点）| 
                    🟢→🔴 waypoints（近→远）| 
                    ➤ 朝向箭头
                </div>
            </body>
            </html>
            '''
        
        @app.route('/video_feed')
        def video_feed():
            return Response(
                streamer._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        print(f"[Client] HTTP 视频流启动: http://0.0.0.0:{self.http_port}")
        app.run(host='0.0.0.0', port=self.http_port, threaded=True)
    
    def _ros2_spin_thread(self):
        """ROS2 spin 线程"""
        try:
            rclpy.spin(self._ros2_node)
        except Exception as e:
            print(f"[Client] ROS2 spin 错误: {e}")
        finally:
            if self._ros2_node:
                self._ros2_node.destroy_node()
            if self._rclpy_initialized:
                rclpy.shutdown()
    
    def run(self, instruction: str = "Follow the person", display: bool = True, http_stream: bool = False):
        """主循环"""
        if not self.connect():
            return
        
        if not self.open_camera():
            return
        
        # 启动 ROS2 节点
        if self.enable_ros2:
            try:
                if not self._rclpy_initialized:
                    rclpy.init()
                    self._rclpy_initialized = True
                self._ros2_node = MotionPublisher()
                self._ros2_thread = Thread(target=self._ros2_spin_thread, daemon=True)
                self._ros2_thread.start()
                print("[Client] ROS2 运动控制节点已启动，发布到 /motion_ref")
                time.sleep(0.5)  # 等待节点初始化
            except Exception as e:
                print(f"[Client] ROS2 启动失败: {e}")
                self.enable_ros2 = False
        
        # 启动 HTTP 流服务器
        if http_stream:
            http_thread = Thread(target=self._start_http_server, daemon=True)
            http_thread.start()
            time.sleep(1)  # 等待服务器启动
        
        print(f"[Client] 开始推理循环，指令: '{instruction}'")
        if display:
            print("[Client] 按 'q' 退出, 'r' 重置历史")
        else:
            print("[Client] 按 Ctrl+C 退出")
        
        frame_interval = 1.0 / self.fps
        last_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Client] 读取帧失败")
                    continue
                
                # 控制帧率
                current_time = time.time()
                if current_time - last_time < frame_interval:
                    continue
                last_time = current_time
                
                # 推理
                start = time.time()
                result = self.infer(frame, instruction)
                rtt = (time.time() - start) * 1000
                
                if result[0] is not None:
                    waypoints, server_time = result
                    vis = self.draw_trajectory(frame, waypoints)
                    
                    # 发布速度命令到 ROS2
                    # 注意：使用 waypoints[1]（第 2 个 waypoint），与 trained_agent.py 的 _planner_action 一致
                    if self.enable_ros2 and self._ros2_node is not None and len(waypoints) > 1:
                        x, y, theta = waypoints[1]
                        # waypoints 是累积位移，转换为速度：速度 = 位移 / 时间尺度
                        # dt = 0.1 秒，与 trained_agent.py 中的 dt 一致
                        linear_x = x / self.vel_time_scale
                        linear_y = y / self.vel_time_scale
                        angular_z = theta / self.vel_time_scale
                        self._ros2_node.update_velocity(linear_x, linear_y, angular_z)
                    
                    # 显示信息
                    info = f"RTT: {rtt:.0f}ms | Server: {server_time:.0f}ms | Frame: {self.frame_id}"
                    cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 终端输出
                    if not display:
                        x, y, theta = waypoints[0] if len(waypoints) > 0 else (0, 0, 0)
                        print(f"[Client] Frame {self.frame_id}: RTT={rtt:.0f}ms, x={x:.3f}m, y={y:.3f}m, theta={math.degrees(theta):.1f}deg")
                else:
                    vis = frame
                    cv2.putText(vis, "Inference Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 更新 HTTP 流
                if http_stream:
                    self._update_http_frame(vis)
                
                # 本地显示
                if display:
                    cv2.imshow('OpenTrackVLA Streamer', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        print("[Client] 发送重置请求...")
                        
        except KeyboardInterrupt:
            print("\n[Client] 收到中断信号")
        finally:
            self.close(display)
    
    def close(self, display: bool = True):
        """清理资源"""
        # 停止速度命令（发送零速度）
        if self.enable_ros2 and self._ros2_node is not None:
            self._ros2_node.update_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)  # 确保最后一条消息发送
        
        if self.cap:
            self.cap.release()
        if self.channel:
            self.channel.close()
        # 只有在显示模式下才调用 destroyAllWindows
        if display:
            cv2.destroyAllWindows()
        print("[Client] 已关闭")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AGX Orin 摄像头采集端')
    parser.add_argument('--server', type=str, default='localhost:50100', help='gRPC服务器地址')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--width', type=int, default=640, help='图像宽度')
    parser.add_argument('--height', type=int, default=480, help='图像高度')
    parser.add_argument('--fps', type=int, default=10, help='目标帧率')
    parser.add_argument('--quality', type=int, default=80, help='JPEG质量 (1-100)')
    parser.add_argument('--instruction', type=str, default='Follow the person', help='文本指令')
    parser.add_argument('--no-display', action='store_true', help='不显示画面（headless模式）')
    parser.add_argument('--http-stream', action='store_true', help='启用HTTP视频流（用于远程查看）')
    parser.add_argument('--http-port', type=int, default=8080, help='HTTP流端口')
    parser.add_argument('--no-ros2', action='store_true', help='禁用ROS2运动控制')
    parser.add_argument('--vel-time-scale', type=float, default=0.1, help='waypoint位移到速度的转换时间尺度（秒），默认0.1秒与trained_agent.py一致')
    
    args = parser.parse_args()
    
    streamer = CameraStreamer(
        server_addr=args.server,
        camera_idx=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.quality,
        http_port=args.http_port,
        enable_ros2=not args.no_ros2,
        vel_time_scale=args.vel_time_scale
    )
    
    streamer.run(
        instruction=args.instruction, 
        display=not args.no_display,
        http_stream=args.http_stream
    )
