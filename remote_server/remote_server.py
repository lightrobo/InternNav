#!/usr/bin/env python3
"""
云端推理服务器 - 基于 InternNav AgentServer

使用 InternNav 原生的 Agent 注册机制和 HTTP 服务器。
支持 VLN (Vision-Language Navigation) 推理。

启动方式:
    python remote_server.py --port 8087

客户端调用:
    1. POST /agent/init  - 初始化 Agent
    2. POST /agent/{name}/step - 单步推理
    3. POST /agent/{name}/reset - 重置状态
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# 导入 Agent 基类和 VLNAgent（触发注册）
from internnav.agent import Agent  # noqa: F401
from internnav.utils import AgentServer

# 导入 VLNAgent 以触发注册
import vln_agent  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description='VLN 云端推理服务器 (基于 InternNav)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器监听地址')
    parser.add_argument('--port', type=int, default=50052, help='服务器端口')
    parser.add_argument('--reload', action='store_true', help='开发模式：自动重载')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLN 云端推理服务器")
    print("=" * 60)
    print(f"监听地址: {args.host}:{args.port}")
    print(f"已注册的 Agent 类型: {list(Agent.agents.keys())}")
    print("=" * 60)
    print()
    print("API 端点:")
    print(f"  POST http://{args.host}:{args.port}/agent/init")
    print(f"       初始化 Agent（body: InitRequest）")
    print(f"  POST http://{args.host}:{args.port}/agent/{{name}}/step")
    print(f"       单步推理（body: StepRequest）")
    print(f"  POST http://{args.host}:{args.port}/agent/{{name}}/reset")
    print(f"       重置状态（body: ResetRequest）")
    print()
    print("示例 - 初始化 VLN Agent:")
    print('''
    curl -X POST http://localhost:8087/agent/init \\
      -H "Content-Type: application/json" \\
      -d '{
        "agent_config": {
          "model_name": "vln",
          "model_settings": {
            "model_dir": "/path/to/model",
            "device": "cuda"
          }
        }
      }'
    ''')
    print("=" * 60)
    
    # 启动服务器
    server = AgentServer(args.host, args.port)
    server.run(reload=args.reload)


if __name__ == '__main__':
    main()
