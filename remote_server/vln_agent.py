#!/usr/bin/env python3
"""
VLN Agent - 适配 InternNav Agent 接口的 VLN 推理 Agent

基于 InternNav 的 Agent 注册机制，复用现有的模型推理逻辑。
返回 waypoints（轨迹点），而非离散动作。
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# 添加项目根目录
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg


@Agent.register('vln')
class VLNAgent(Agent):
    """
    VLN (Vision-Language Navigation) Agent
    
    输入 obs:
        - 'rgb': RGB 图像 (H, W, 3) numpy array 或 PIL.Image
        - 'instruction': 文本指令 (str)
        - 'depth': 深度图 (可选)
    
    输出 action:
        - 'waypoints': 轨迹点 (N, 3) - [x, y, theta]
        - 'n_waypoints': 轨迹点数量
        - 'inference_time_ms': 推理耗时
        - 'success': 是否成功
    """

    def __init__(self, config: AgentCfg):
        super().__init__(config)
        
        model_settings = config.model_settings or {}
        self.model_dir = model_settings.get('model_dir') or os.environ.get('HF_MODEL_DIR')
        self.device = model_settings.get('device', 'cuda')
        
        # 延迟加载模型（首次 step 时加载）
        self._model = None
        self._last_predicted_traj = None
        
        print(f"[VLNAgent] 初始化完成")
        print(f"[VLNAgent] 模型目录: {self.model_dir}")
        print(f"[VLNAgent] 设备: {self.device}")

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return
        
        print(f"[VLNAgent] 加载模型...")
        
        # 设置环境变量
        if self.model_dir:
            os.environ['HF_MODEL_DIR'] = self.model_dir
        
        # 尝试导入模型
        # 方式1: 使用 InternNav 的 policy 系统
        try:
            from internnav.configs.model.base_encoders import ModelCfg
            from internnav.model import get_config, get_policy
            
            model_settings = self.config.model_settings or {}
            policy_name = model_settings.get('policy_name', 'internvla_n1')
            
            policy = get_policy(policy_name)
            policy_config = get_config(policy_name)
            self._model = policy(config=policy_config(model_cfg={'model': model_settings}))
            self._model.eval()
            self._model_type = 'internnav'
            print(f"[VLNAgent] 使用 InternNav policy: {policy_name}")
            return
        except Exception as e:
            print(f"[VLNAgent] InternNav policy 加载失败: {e}")
        
        # 方式2: 尝试加载 GTBBoxAgent（兼容旧代码）
        try:
            # 假设 trained_agent.py 在某个位置
            from trained_agent import GTBBoxAgent
            
            tmp_dir = "/tmp/vln_agent"
            os.makedirs(tmp_dir, exist_ok=True)
            self._model = GTBBoxAgent(result_path=tmp_dir)
            self._model_type = 'gtbbox'
            print(f"[VLNAgent] 使用 GTBBoxAgent")
            return
        except Exception as e:
            print(f"[VLNAgent] GTBBoxAgent 加载失败: {e}")
        
        raise RuntimeError("无法加载任何模型，请检查配置")

    def step(self, obs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行单步推理
        
        Args:
            obs: 观测数据列表，每个元素是一个 dict:
                - 'rgb': RGB 图像
                - 'instruction': 文本指令
                - 'depth': 深度图 (可选)
        
        Returns:
            action 列表，每个元素是一个 dict:
                - 'waypoints': 轨迹点列表 [[x, y, theta], ...]
                - 'n_waypoints': 轨迹点数量
                - 'inference_time_ms': 推理耗时
                - 'success': 是否成功
                - 'error': 错误信息 (如果失败)
        """
        # 确保模型已加载
        self._load_model()
        
        results = []
        for single_obs in obs:
            result = self._step_single(single_obs)
            results.append(result)
        
        return results

    def _step_single(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """单个观测的推理"""
        start_time = time.time()
        
        try:
            # 提取输入
            rgb = obs.get('rgb')
            instruction = obs.get('instruction', 'follow the person')
            depth = obs.get('depth')
            
            # 处理图像格式
            if isinstance(rgb, bytes):
                # JPEG 编码的图像
                import io
                rgb = np.array(Image.open(io.BytesIO(rgb)).convert('RGB'))
            elif isinstance(rgb, Image.Image):
                rgb = np.array(rgb.convert('RGB'))
            elif isinstance(rgb, np.ndarray):
                # 确保是 RGB 格式
                if rgb.shape[-1] == 4:  # RGBA
                    rgb = rgb[:, :, :3]
            
            # 调用模型推理
            if self._model_type == 'internnav':
                # InternNav policy 推理
                action = self._infer_internnav(rgb, instruction, depth)
            else:
                # GTBBoxAgent 推理
                action = self._infer_gtbbox(rgb, instruction)
            
            inference_time = (time.time() - start_time) * 1000
            
            if action is None:
                return {
                    'waypoints': [],
                    'n_waypoints': 0,
                    'inference_time_ms': inference_time,
                    'success': False,
                    'error': 'Model returned None'
                }
            
            # 获取轨迹
            traj = self._last_predicted_traj
            if traj is not None:
                waypoints = traj.tolist()
                n_waypoints = len(waypoints)
            else:
                # 如果没有轨迹，返回 action 作为单个 waypoint
                waypoints = [action] if isinstance(action, list) else action.tolist()
                n_waypoints = 1
            
            return {
                'waypoints': waypoints,
                'n_waypoints': n_waypoints,
                'inference_time_ms': inference_time,
                'success': True
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'waypoints': [],
                'n_waypoints': 0,
                'inference_time_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error': str(e)
            }

    def _infer_internnav(self, rgb: np.ndarray, instruction: str, depth: Optional[np.ndarray] = None):
        """使用 InternNav policy 推理"""
        # 构造 obs 格式
        policy_obs = {
            'rgb': rgb,
            'instruction': instruction,
        }
        if depth is not None:
            policy_obs['depth'] = depth
        
        # 调用 policy
        result = self._model.step([policy_obs])
        
        # 提取轨迹
        if hasattr(self._model, '_last_predicted_traj'):
            self._last_predicted_traj = self._model._last_predicted_traj
        
        return result

    def _infer_gtbbox(self, rgb: np.ndarray, instruction: str):
        """使用 GTBBoxAgent 推理"""
        action = self._model._planner_action(rgb, instruction)
        
        # 获取预测的轨迹
        if hasattr(self._model, '_last_predicted_traj'):
            self._last_predicted_traj = self._model._last_predicted_traj
        
        return action

    def reset(self, reset_index=None):
        """重置 Agent 状态"""
        self._last_predicted_traj = None
        
        if self._model is not None:
            # 重置模型内部状态
            if hasattr(self._model, 'reset'):
                self._model.reset()
            elif hasattr(self._model, '_coarse_hist_tokens'):
                self._model._coarse_hist_tokens.clear()
        
        print(f"[VLNAgent] 状态已重置")


# 为了让 Agent 被注册，需要在导入时执行
__all__ = ['VLNAgent']
