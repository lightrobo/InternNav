#!/usr/bin/env python3
"""
gRPC 推理服务器 - 适配 camera_streamer 的 gRPC 客户端

直接使用 InternVLAN1AsyncAgent（实机部署版本）进行推理。

启动方式:
    python grpc_server.py --port 50052 --model-path /path/to/model
"""

import argparse
import io
import os
import sys
import time
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import cv2
import grpc
import numpy as np
from PIL import Image

# 添加项目根目录
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

# 导入生成的 gRPC 代码
import inference_pb2
import inference_pb2_grpc


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC InferenceService 实现 - 使用 InternVLAN1AsyncAgent"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.model_path = model_path or os.environ.get('HF_MODEL_DIR')
        self.device = device
        self._agent = None
        self._initialized = False
        
        # Agent 配置参数
        self.resize_w = 384
        self.resize_h = 384
        self.num_history = 4
        self.plan_step_gap = 8
        
        print(f"[gRPC] 服务初始化")
        print(f"[gRPC] 模型路径: {self.model_path or '(未设置)'}")
        print(f"[gRPC] 设备: {self.device}")
        
        # 如果有模型路径，立即加载
        if self.model_path:
            self._load_agent()
    
    def _load_agent(self):
        """加载 InternVLAN1AsyncAgent"""
        if self._agent is not None:
            return
        
        if not self.model_path:
            raise RuntimeError("模型路径未设置！请使用 --model-path 参数或设置 HF_MODEL_DIR 环境变量")
        
        print(f"[gRPC] 加载 InternVLAN1AsyncAgent...")
        print(f"[gRPC] 模型路径: {self.model_path}")
        
        # 导入 Agent
        from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
        
        # 构造参数
        args = SimpleNamespace(
            model_path=self.model_path,
            device=self.device,
            resize_w=self.resize_w,
            resize_h=self.resize_h,
            num_history=self.num_history,
            plan_step_gap=self.plan_step_gap,
        )
        
        self._agent = InternVLAN1AsyncAgent(args)
        self._initialized = True
        print(f"[gRPC] Agent 加载完成")
    
    def _decode_image(self, image_data: bytes) -> np.ndarray:
        """解码 JPEG 图像"""
        img = Image.open(io.BytesIO(image_data))
        return np.array(img.convert('RGB'))
    
    def _decode_depth(self, depth_data: bytes, width: int, height: int) -> np.ndarray:
        """解码 PNG 深度图"""
        if not depth_data:
            return None
        
        # PNG 解码
        depth_array = np.frombuffer(depth_data, dtype=np.uint8)
        depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        
        if depth is None:
            print("[gRPC] 警告: 深度图解码失败")
            return None
        
        return depth
    
    def _get_default_intrinsic(self, width: int, height: int, hfov: float = 69.4) -> np.ndarray:
        """获取默认相机内参（D435 默认）"""
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        
        intrinsic = np.array([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        return intrinsic
    
    def Infer(self, request, context):
        """单帧推理"""
        start_time = time.time()
        
        try:
            # 确保 Agent 已加载
            self._load_agent()
            
            # 解码输入
            rgb = self._decode_image(request.image_data)
            depth = self._decode_depth(
                request.depth_data, 
                request.depth_width, 
                request.depth_height
            )
            instruction = request.instruction or "follow the person"
            
            # 默认位姿（单位矩阵）
            pose = np.eye(4)
            
            # 相机内参
            h, w = rgb.shape[:2]
            intrinsic = self._get_default_intrinsic(w, h)
            
            # 处理深度图格式
            if depth is not None:
                # 转换深度图为 Agent 期望的格式
                # D435 深度图是 uint16，单位是 mm，需要转换
                if depth.dtype == np.uint16:
                    # 转换为 uint8 用于可视化
                    depth_scale = request.depth_scale if request.depth_scale > 0 else 0.001
                    depth_m = depth.astype(np.float32) * depth_scale
                    depth_vis = (np.clip(depth_m / 5.0 * 255, 0, 255)).astype(np.uint8)
                else:
                    depth_vis = depth
            else:
                # 如果没有深度图，创建一个空的
                depth_vis = np.zeros((h, w), dtype=np.uint8)
            
            # 调用 Agent 推理
            output = self._agent.step(rgb, depth_vis, pose, instruction, intrinsic)
            
            inference_time = (time.time() - start_time) * 1000
            
            # 处理输出
            waypoints = []
            n_waypoints = 0
            
            if output.output_trajectory is not None:
                # 轨迹格式：[[x, y, theta], ...]
                traj = output.output_trajectory
                if isinstance(traj, np.ndarray):
                    traj = traj.tolist()
                
                # 展平 waypoints
                for wp in traj:
                    if isinstance(wp, (list, np.ndarray)):
                        waypoints.extend([float(v) for v in wp[:3]])  # x, y, theta
                    else:
                        waypoints.append(float(wp))
                
                n_waypoints = len(traj)
                
            elif output.output_action is not None:
                # 如果是离散动作，转换为简单的 waypoint
                # 动作: 0=STOP, 1=前进, 2=左转, 3=右转, 5=低头
                actions = output.output_action
                for action in actions:
                    if action == 0:  # STOP
                        waypoints.extend([0.0, 0.0, 0.0])
                    elif action == 1:  # 前进
                        waypoints.extend([0.25, 0.0, 0.0])
                    elif action == 2:  # 左转
                        waypoints.extend([0.0, 0.0, 0.26])  # ~15度
                    elif action == 3:  # 右转
                        waypoints.extend([0.0, 0.0, -0.26])
                    else:
                        waypoints.extend([0.0, 0.0, 0.0])
                
                n_waypoints = len(actions)
            
            print(f"[gRPC] 推理完成: {n_waypoints} waypoints, {inference_time:.0f}ms")
            
            return inference_pb2.InferResponse(
                frame_id=request.frame_id,
                waypoints=waypoints,
                n_waypoints=n_waypoints,
                inference_time_ms=inference_time,
                success=True,
            )
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return inference_pb2.InferResponse(
                frame_id=request.frame_id,
                inference_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )
    
    def StreamInfer(self, request_iterator, context):
        """流式推理（双向流）"""
        for request in request_iterator:
            yield self.Infer(request, context)
    
    def HealthCheck(self, request, context):
        """健康检查"""
        return inference_pb2.HealthResponse(
            healthy=True,
            model_status='loaded' if self._initialized else 'not_loaded',
            device=self.device,
        )


def serve(host: str, port: int, model_path: str = None, device: str = 'cuda'):
    """启动 gRPC 服务器"""
    
    # 创建服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    # 注册服务
    servicer = InferenceServicer(model_path=model_path, device=device)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    
    # 绑定地址
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    
    # 启动
    server.start()
    
    print("=" * 60)
    print("gRPC 推理服务器 (InternVLAN1AsyncAgent)")
    print("=" * 60)
    print(f"监听地址: {address}")
    print(f"模型路径: {model_path or '(未设置)'}")
    print(f"设备: {device}")
    print("=" * 60)
    print()
    print("gRPC 方法:")
    print(f"  Infer(InferRequest) -> InferResponse")
    print(f"  StreamInfer(stream InferRequest) -> stream InferResponse")
    print(f"  HealthCheck(Empty) -> HealthResponse")
    print()
    print("客户端连接:")
    print(f"  grpc.insecure_channel('{address}')")
    print("=" * 60)
    print()
    print("按 Ctrl+C 退出")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[gRPC] 收到中断信号，正在关闭...")
        server.stop(grace=5)
        print("[gRPC] 服务器已关闭")


def main():
    # 设置项目路径
    print(f"PROJECT_ROOT_PATH:{project_root}")
    
    parser = argparse.ArgumentParser(description='gRPC 推理服务器 (InternVLAN1AsyncAgent)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=50052, help='监听端口')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='InternVLA-N1 模型路径 (HuggingFace 格式)')
    parser.add_argument('--device', type=str, default='cuda', help='推理设备')
    
    args = parser.parse_args()
    
    # 优先使用命令行参数，其次环境变量
    model_path = args.model_path or os.environ.get('HF_MODEL_DIR')
    
    serve(
        host=args.host,
        port=args.port,
        model_path=model_path,
        device=args.device,
    )


if __name__ == '__main__':
    main()
