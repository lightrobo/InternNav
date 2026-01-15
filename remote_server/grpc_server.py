#!/usr/bin/env python3
"""
gRPC 推理服务器 - 适配 camera_streamer 的 gRPC 客户端

实现 inference.proto 定义的 InferenceService：
- Infer: 单帧推理（RGBD + instruction → waypoints）
- HealthCheck: 健康检查
- StreamInfer: 流式推理（可选）

启动方式:
    python grpc_server.py --port 50052
"""

import argparse
import io
import os
import sys
import time
from concurrent import futures
from pathlib import Path

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

# 导入 Agent
from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg

# 导入 VLNAgent（触发注册）
import vln_agent  # noqa: F401


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC InferenceService 实现"""
    
    def __init__(self, model_dir: str = None, device: str = 'cuda'):
        self.model_dir = model_dir or os.environ.get('HF_MODEL_DIR')
        self.device = device
        self._agent = None
        self._initialized = False
        
        print(f"[gRPC] 服务初始化")
        print(f"[gRPC] 模型目录: {self.model_dir or '(将在首次推理时指定)'}")
        print(f"[gRPC] 设备: {self.device}")
    
    def _ensure_agent(self):
        """确保 Agent 已初始化"""
        if self._agent is not None:
            return
        
        print("[gRPC] 初始化 VLN Agent...")
        
        config = AgentCfg(
            model_name='vln',
            model_settings={
                'model_dir': self.model_dir,
                'device': self.device,
            }
        )
        
        self._agent = Agent.init(config)
        self._initialized = True
        print("[gRPC] VLN Agent 初始化完成")
    
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
    
    def Infer(self, request, context):
        """单帧推理"""
        start_time = time.time()
        
        try:
            # 确保 Agent 已初始化
            self._ensure_agent()
            
            # 解码输入
            rgb = self._decode_image(request.image_data)
            depth = self._decode_depth(
                request.depth_data, 
                request.depth_width, 
                request.depth_height
            )
            instruction = request.instruction or "follow the person"
            
            # 构造观测
            obs = {
                'rgb': rgb,
                'instruction': instruction,
            }
            if depth is not None:
                obs['depth'] = depth
                obs['depth_scale'] = request.depth_scale
            
            # 调用 Agent 推理
            results = self._agent.step([obs])
            result = results[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            if result.get('success', False):
                waypoints = result.get('waypoints', [])
                n_waypoints = result.get('n_waypoints', len(waypoints))
                
                # 展平 waypoints: [[x, y, theta], ...] -> [x, y, theta, x, y, theta, ...]
                flat_waypoints = []
                for wp in waypoints:
                    flat_waypoints.extend(wp)
                
                return inference_pb2.InferResponse(
                    frame_id=request.frame_id,
                    waypoints=flat_waypoints,
                    n_waypoints=n_waypoints,
                    inference_time_ms=inference_time,
                    success=True,
                )
            else:
                return inference_pb2.InferResponse(
                    frame_id=request.frame_id,
                    inference_time_ms=inference_time,
                    success=False,
                    error=result.get('error', 'Unknown error'),
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


def serve(host: str, port: int, model_dir: str = None, device: str = 'cuda'):
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
    servicer = InferenceServicer(model_dir=model_dir, device=device)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    
    # 绑定地址
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    
    # 启动
    server.start()
    
    print("=" * 60)
    print("gRPC 推理服务器")
    print("=" * 60)
    print(f"监听地址: {address}")
    print(f"已注册的 Agent 类型: {list(Agent.agents.keys())}")
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
    
    parser = argparse.ArgumentParser(description='gRPC 推理服务器')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=50052, help='监听端口')
    parser.add_argument('--model-dir', type=str, default=None, help='模型目录')
    parser.add_argument('--device', type=str, default='cuda', help='推理设备')
    
    args = parser.parse_args()
    
    serve(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        device=args.device,
    )


if __name__ == '__main__':
    main()
