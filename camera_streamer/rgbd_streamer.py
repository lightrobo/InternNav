#!/usr/bin/env python3
"""
AGX Orin RealSense D435 RGBD采集端 - gRPC Client
采集D435的RGB+Depth画面，通过gRPC发送到云端推理
支持 HTTP 视频流用于远程查看（RGB + Depth colormap并排显示）

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

# RealSense
import pyrealsense2 as rs

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc


class RGBDStreamer:
    """RealSense D435 RGBD采集 + gRPC 客户端 + HTTP 视频流"""
    
    def __init__(
        self,
        server_addr: str = "localhost:50051",
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        jpeg_quality: int = 80,
        http_port: int = 8080,
        align_depth: bool = True,
    ):
        self.server_addr = server_addr
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.http_port = http_port
        self.align_depth = align_depth
        
        self.channel = None
        self.stub = None
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = 0.001  # D435默认depth scale
        self.frame_id = 0
        
        # HTTP 流相关
        self._current_frame = None
        self._frame_lock = Lock()
        self._http_server = None
        
    def connect(self) -> bool:
        """连接gRPC服务器"""
        print(f"[RGBD] 连接服务器: {self.server_addr}")
        
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
            print(f"[RGBD] 服务器状态: healthy={response.healthy}, device={response.device}")
            return response.healthy
        except grpc.RpcError as e:
            print(f"[RGBD] 连接失败: {e}")
            return False
    
    def open_camera(self) -> bool:
        """打开RealSense D435"""
        print(f"[RGBD] 初始化 RealSense D435...")
        
        try:
            # 创建pipeline和config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # 检测可用设备
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("[RGBD] 未检测到RealSense设备!")
                return False
            
            device = devices[0]
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"[RGBD] 检测到设备: {name} (SN: {serial})")
            
            # 配置RGB和Depth流
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # 启动pipeline
            profile = self.pipeline.start(self.config)
            
            # 获取深度比例因子
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"[RGBD] 深度比例因子: {self.depth_scale} (depth_value * scale = meters)")
            
            # 创建对齐器（将深度图对齐到RGB）
            if self.align_depth:
                self.align = rs.align(rs.stream.color)
                print("[RGBD] 深度图对齐到RGB已启用")
            
            # 获取实际参数
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            
            print(f"[RGBD] RGB流: {color_profile.width()}x{color_profile.height()} @ {color_profile.fps()}fps")
            print(f"[RGBD] Depth流: {depth_profile.width()}x{depth_profile.height()} @ {depth_profile.fps()}fps")
            
            # 预热（丢弃前几帧让自动曝光稳定）
            print("[RGBD] 预热中...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            print("[RGBD] 预热完成")
            
            return True
            
        except Exception as e:
            print(f"[RGBD] 初始化失败: {e}")
            return False
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取对齐的RGB和Depth帧"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            if self.align:
                frames = self.align.process(frames)
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"[RGBD] 获取帧失败: {e}")
            return None, None
    
    def encode_rgb(self, frame: np.ndarray) -> bytes:
        """JPEG编码RGB图像"""
        # BGR -> RGB for server
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buffer.tobytes()
    
    def encode_depth(self, depth: np.ndarray) -> bytes:
        """PNG编码深度图（16bit无损）"""
        _, buffer = cv2.imencode('.png', depth)
        return buffer.tobytes()
    
    def depth_to_colormap(self, depth: np.ndarray, max_depth_m: float = 5.0) -> np.ndarray:
        """将深度图转换为colormap用于可视化"""
        # 转换为米
        depth_m = depth.astype(np.float32) * self.depth_scale
        # 归一化到0-255
        depth_normalized = np.clip(depth_m / max_depth_m * 255, 0, 255).astype(np.uint8)
        # 应用colormap
        colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return colormap
    
    def infer(self, color: np.ndarray, depth: np.ndarray, instruction: str) -> Tuple[Optional[np.ndarray], float]:
        """发送RGBD帧到云端推理，返回 waypoints（累积位移序列）"""
        self.frame_id += 1
        
        # 编码
        image_data = self.encode_rgb(color)
        depth_data = self.encode_depth(depth)
        
        # 发送请求
        request = inference_pb2.InferRequest(
            image_data=image_data,
            instruction=instruction,
            frame_id=self.frame_id,
            timestamp_ms=int(time.time() * 1000),
            depth_data=depth_data,
            depth_width=depth.shape[1],
            depth_height=depth.shape[0],
            depth_scale=self.depth_scale,
        )
        
        try:
            response = self.stub.Infer(request)
            
            if response.success:
                waypoints = np.array(response.waypoints).reshape(response.n_waypoints, 3)
                return waypoints, response.inference_time_ms
            else:
                print(f"[RGBD] 推理错误: {response.error}")
                return None, 0
        except grpc.RpcError as e:
            print(f"[RGBD] gRPC错误: {e}")
            return None, 0
    
    def draw_trajectory(self, frame: np.ndarray, waypoints: np.ndarray, scale: float = 120.0) -> np.ndarray:
        """在画面上绘制轨迹"""
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx = w // 2
        cy = int(h * 0.86)
        
        arrow_len = 20
        
        points = [(cx, cy)]
        for i, (x, y, theta) in enumerate(waypoints):
            px = int(cx - y * scale)
            py = int(cy - x * scale)
            points.append((px, py))
            
            progress = i / max(len(waypoints) - 1, 1)
            color = (
                int(255 * progress),
                int(255 * (1 - progress)),
                0
            )
            cv2.circle(vis, (px, py), 5, color, -1)
            cv2.circle(vis, (px, py), 7, (255, 255, 255), 1)
            
            arrow_dx = int(-arrow_len * math.sin(theta))
            arrow_dy = int(-arrow_len * math.cos(theta))
            arrow_end = (px + arrow_dx, py + arrow_dy)
            
            turn_intensity = min(abs(theta) / 0.5, 1.0)
            arrow_color = (
                int(255 * turn_intensity),
                int(255 * (1 - turn_intensity * 0.5)),
                0
            )
            cv2.arrowedLine(vis, (px, py), arrow_end, arrow_color, 2, tipLength=0.4)
        
        for i in range(len(points) - 1):
            progress = i / max(len(points) - 2, 1)
            line_color = (
                int(100 * progress),
                int(200 * (1 - progress * 0.5)),
                0
            )
            cv2.line(vis, points[i], points[i + 1], line_color, 2)
        
        cv2.circle(vis, (cx, cy), 12, (255, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 14, (255, 255, 255), 2)
        cv2.arrowedLine(vis, (cx, cy), (cx, cy - 35), (255, 100, 100), 3, tipLength=0.3)
        cv2.putText(vis, "Robot", (cx - 25, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(waypoints) > 0:
            x, y, theta = waypoints[0]
            wp_info = f"WP0: x={x:.2f}m y={y:.2f}m th={math.degrees(theta):.1f}deg"
            cv2.putText(vis, wp_info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return vis
    
    def create_combined_view(self, color: np.ndarray, depth: np.ndarray, waypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """创建RGB + Depth colormap并排视图"""
        # 在RGB上绘制轨迹
        if waypoints is not None:
            color_vis = self.draw_trajectory(color, waypoints)
        else:
            color_vis = color.copy()
        
        # Depth colormap
        depth_colormap = self.depth_to_colormap(depth)
        
        # 添加标签
        cv2.putText(color_vis, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(depth_colormap, "Depth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 水平拼接
        combined = np.hstack([color_vis, depth_colormap])
        
        return combined
    
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
            
            time.sleep(0.033)
    
    def _start_http_server(self):
        """启动 HTTP 视频流服务器"""
        try:
            from flask import Flask, Response
        except ImportError:
            print("[RGBD] 警告: Flask 未安装，HTTP 流功能不可用")
            print("[RGBD] 安装: pip install flask")
            return
        
        app = Flask(__name__)
        streamer = self
        
        @app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>RGBD Live Stream</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
                        color: #e0e0ff; 
                        font-family: 'JetBrains Mono', 'Fira Code', monospace;
                        min-height: 100vh;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        padding: 20px;
                    }
                    .header {
                        display: flex;
                        align-items: center;
                        gap: 15px;
                        margin-bottom: 20px;
                    }
                    h1 { 
                        color: #00ffaa;
                        font-size: 1.8em;
                        font-weight: 600;
                        text-shadow: 0 0 20px rgba(0, 255, 170, 0.3);
                    }
                    .status-badge {
                        background: linear-gradient(90deg, #00ff88, #00ccff);
                        color: #000;
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 0.75em;
                        font-weight: bold;
                        animation: pulse 2s infinite;
                    }
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.7; }
                    }
                    .video-container {
                        position: relative;
                        border: 3px solid #00ffaa;
                        border-radius: 12px;
                        overflow: hidden;
                        box-shadow: 0 0 40px rgba(0, 255, 170, 0.2);
                    }
                    img { 
                        display: block;
                        max-width: 100%;
                        height: auto;
                    }
                    .info-panel {
                        margin-top: 20px;
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 15px;
                        max-width: 800px;
                    }
                    .info-card {
                        background: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(0, 255, 170, 0.3);
                        border-radius: 8px;
                        padding: 15px;
                    }
                    .info-card h3 {
                        color: #00ffaa;
                        font-size: 0.9em;
                        margin-bottom: 8px;
                    }
                    .info-card p {
                        color: #888;
                        font-size: 0.8em;
                        line-height: 1.6;
                    }
                    .legend {
                        margin-top: 15px;
                        padding: 15px;
                        background: rgba(255, 255, 255, 0.03);
                        border-radius: 8px;
                        font-size: 0.8em;
                        color: #aaa;
                        max-width: 800px;
                    }
                    .legend-item {
                        display: inline-block;
                        margin-right: 20px;
                    }
                    .legend-color {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        margin-right: 5px;
                        vertical-align: middle;
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📷 RGBD Live Stream</h1>
                    <span class="status-badge">● LIVE</span>
                </div>
                
                <div class="video-container">
                    <img src="/video_feed" alt="RGBD Stream">
                </div>
                
                <div class="info-panel">
                    <div class="info-card">
                        <h3>🎨 RGB View</h3>
                        <p>左侧显示RGB彩色图像，叠加预测轨迹</p>
                    </div>
                    <div class="info-card">
                        <h3>📏 Depth View</h3>
                        <p>右侧显示深度图colormap<br>
                        蓝色=近 → 红色=远</p>
                    </div>
                </div>
                
                <div class="legend">
                    <span class="legend-item"><span class="legend-color" style="background: #ff0000;"></span>机器人位置</span>
                    <span class="legend-item"><span class="legend-color" style="background: #00ff00;"></span>近处waypoint</span>
                    <span class="legend-item"><span class="legend-color" style="background: #ff6600;"></span>远处waypoint</span>
                    <span class="legend-item">➤ 朝向箭头</span>
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
        
        print(f"[RGBD] HTTP 视频流启动: http://0.0.0.0:{self.http_port}")
        app.run(host='0.0.0.0', port=self.http_port, threaded=True)
    
    def run(self, instruction: str = "Follow the person", display: bool = True, http_stream: bool = False, no_infer: bool = False):
        """主循环
        
        Args:
            instruction: 推理指令
            display: 是否本地显示
            http_stream: 是否启用HTTP流
            no_infer: 仅预览模式（不连接服务器）
        """
        if not no_infer:
            if not self.connect():
                return
        else:
            print("[RGBD] 预览模式：不连接推理服务器")
        
        if not self.open_camera():
            return
        
        # 启动 HTTP 流服务器
        if http_stream:
            http_thread = Thread(target=self._start_http_server, daemon=True)
            http_thread.start()
            time.sleep(1)
        
        print(f"[RGBD] 开始{'推理' if not no_infer else '预览'}循环，指令: '{instruction}'")
        if display:
            print("[RGBD] 按 'q' 退出")
        else:
            print("[RGBD] 按 Ctrl+C 退出")
        
        frame_interval = 1.0 / self.fps
        last_time = time.time()
        
        try:
            while True:
                color, depth = self.get_frames()
                if color is None or depth is None:
                    continue
                
                # 控制帧率
                current_time = time.time()
                if current_time - last_time < frame_interval:
                    continue
                last_time = current_time
                
                if no_infer:
                    # 预览模式：只显示RGB+Depth
                    vis = self.create_combined_view(color, depth, None)
                    info = f"Preview Mode | Frame: {self.frame_id}"
                    self.frame_id += 1
                else:
                    # 推理模式
                    start = time.time()
                    result = self.infer(color, depth, instruction)
                    rtt = (time.time() - start) * 1000
                    
                    if result[0] is not None:
                        waypoints, server_time = result
                        vis = self.create_combined_view(color, depth, waypoints)
                        info = f"RTT: {rtt:.0f}ms | Server: {server_time:.0f}ms | Frame: {self.frame_id}"
                        
                        if not display:
                            x, y, theta = waypoints[0] if len(waypoints) > 0 else (0, 0, 0)
                            print(f"[RGBD] Frame {self.frame_id}: RTT={rtt:.0f}ms, x={x:.3f}m, y={y:.3f}m, theta={math.degrees(theta):.1f}deg")
                    else:
                        vis = self.create_combined_view(color, depth, None)
                        info = "Inference Failed"
                
                cv2.putText(vis, info, (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 更新 HTTP 流
                if http_stream:
                    self._update_http_frame(vis)
                
                # 本地显示
                if display:
                    cv2.imshow('RGBD Streamer', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\n[RGBD] 收到中断信号")
        finally:
            self.close(display)
    
    def close(self, display: bool = True):
        """清理资源"""
        if self.pipeline:
            self.pipeline.stop()
        if self.channel:
            self.channel.close()
        if display:
            cv2.destroyAllWindows()
        print("[RGBD] 已关闭")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AGX Orin RealSense D435 RGBD采集端')
    parser.add_argument('--server', type=str, default='localhost:50051', help='gRPC服务器地址')
    parser.add_argument('--width', type=int, default=640, help='图像宽度')
    parser.add_argument('--height', type=int, default=480, help='图像高度')
    parser.add_argument('--fps', type=int, default=15, help='目标帧率')
    parser.add_argument('--quality', type=int, default=80, help='JPEG质量 (1-100)')
    parser.add_argument('--instruction', type=str, default='Follow the person', help='文本指令')
    parser.add_argument('--no-display', action='store_true', help='不显示画面（headless模式）')
    parser.add_argument('--http-stream', action='store_true', help='启用HTTP视频流')
    parser.add_argument('--http-port', type=int, default=8080, help='HTTP流端口')
    parser.add_argument('--no-align', action='store_true', help='不对齐深度图到RGB')
    parser.add_argument('--no-infer', action='store_true', help='预览模式：不连接服务器')
    
    args = parser.parse_args()
    
    streamer = RGBDStreamer(
        server_addr=args.server,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.quality,
        http_port=args.http_port,
        align_depth=not args.no_align,
    )
    
    streamer.run(
        instruction=args.instruction, 
        display=not args.no_display,
        http_stream=args.http_stream,
        no_infer=args.no_infer,
    )
