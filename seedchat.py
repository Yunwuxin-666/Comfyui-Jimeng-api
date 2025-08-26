#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
豆包大语言模型节点 - 使用火山引擎doubao-seed-1.6模型
支持文本、图像、视频输入的多模态对话
"""

import json
import requests
import logging
import time
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import base64
import cv2
import tempfile
import os

# 设置日志
logger = logging.getLogger(__name__)

class DoubaoLLMNode:
    """
    调用火山引擎豆包大语言模型API进行多模态对话
    支持doubao-seed-1.6模型的深度思考能力
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "placeholder": "请输入您的问题或对话内容"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的火山引擎API Key"
                }),
                "thinking_mode": ([
                    "auto",
                    "thinking",
                    "non-thinking"
                ], {
                    "default": "auto",
                    "tooltip": "思考模式：auto-自动选择，thinking-深度思考，non-thinking-快速响应"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "温度参数，控制输出的随机性"
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "display": "number",
                    "tooltip": "最大输出长度"
                }),
                "stream": ("BOOLEAN", {
                    "default": False,
                    "label_on": "流式输出",
                    "label_off": "非流式输出"
                }),
                "timeout": ("INT", {
                    "default": 60,
                    "min": 30,
                    "max": 600,
                    "step": 10,
                    "display": "number",
                    "tooltip": "请求超时时间（秒）"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "系统提示词（可选）"
                }),
                "conversation_history": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "placeholder": "对话历史（JSON格式）"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("response", "thinking_content", "conversation_json", "total_tokens", "thinking_tokens")
    FUNCTION = "chat"
    CATEGORY = "✨即梦AI生成"
    DESCRIPTION = """使用火山引擎doubao-seed-1.6模型进行多模态对话。
    
模型特性：
- 支持深度思考模式（thinking/non-thinking/auto）
- 支持文本、图像、视频输入
- 最大上下文长度：256k
- 最大输入长度：224k
- 最大思维链长度：32k
- 默认最大输出：4k（可配置到16k）
    
参数说明：
- prompt: 用户输入的问题或对话内容
- api_key: 火山引擎API密钥
- thinking_mode: 思考模式选择
- temperature: 控制输出随机性（0-2）
- max_tokens: 最大输出token数
- stream: 是否使用流式输出
- timeout: 请求超时时间（秒，默300秒）
- image: 可选，输入图像
- video_frames: 可选，视频帧序列
- system_prompt: 可选，系统提示词
- conversation_history: 可选，对话历史
- seed: 随机种子（-1表示随机）

输出说明：
- response: 模型回复内容
- thinking_content: 深度思考过程（如果有）
- conversation_json: 更新后的对话历史
- total_tokens: 总token使用量
- thinking_tokens: 思考过程token数"""

    def tensor_to_base64(self, tensor_image):
        """将张量图像转换为base64字符串"""
        # 确保张量在CPU上并转换为numpy
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]  # 取第一张图片
        
        # 转换为numpy数组
        image_np = tensor_image.cpu().numpy()
        
        # 如果是归一化的值，转换为0-255
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 如果通道在最后一维，不需要转换
        if image_np.shape[-1] != 3:
            # 如果通道在第一维，转换为HWC格式
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # 创建PIL图像
        pil_image = Image.fromarray(image_np)
        
        # 转换为base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"

    def process_video_frames(self, frames_tensor):
        """处理视频帧，提取关键帧并转换为base64"""
        # 如果帧数太多，进行采样
        total_frames = frames_tensor.shape[0]
        max_frames = 8  # 最多发送8帧
        
        if total_frames <= max_frames:
            selected_indices = list(range(total_frames))
        else:
            # 均匀采样
            step = total_frames / max_frames
            selected_indices = [int(i * step) for i in range(max_frames)]
        
        # 转换选中的帧为base64
        frame_data = []
        for idx in selected_indices:
            frame = frames_tensor[idx]
            base64_image = self.tensor_to_base64(frame)
            frame_data.append({
                "frame_index": idx,
                "image": base64_image
            })
        
        return frame_data, len(selected_indices)

    def chat(self, prompt, api_key, thinking_mode, temperature, max_tokens, stream, timeout,
             image=None, video_frames=None, system_prompt="", conversation_history="[]", seed=-1):
        """调用火山引擎API进行对话"""
        
        # 优先读取apikey.txt
        apikey_path = os.path.join(os.path.dirname(__file__), "apikey.txt")
        file_api_key = ""
        if os.path.exists(apikey_path):
            with open(apikey_path, "r", encoding="utf-8") as f:
                file_api_key = f.read().strip()
        
        use_api_key = file_api_key if file_api_key else api_key
        if not use_api_key:
            raise ValueError("请在apikey.txt或前端页面输入有效的API Key")
        api_key = use_api_key
        
        if not prompt and image is None and video_frames is None:
            raise ValueError("请提供对话内容或输入图像/视频")
        
        # API配置
        api_endpoint = "https://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3/chat/completions"
        model_id = "doubao-seed-1-6-250615"  # 支持thinking字段的版本
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 解析对话历史
        try:
            messages = json.loads(conversation_history) if conversation_history else []
        except:
            messages = []
        
        # 添加系统提示词
        if system_prompt:
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
        
        # 构建用户消息
        user_content = []
        
        # 添加文本内容
        if prompt:
            user_content.append({
                "type": "text",
                "text": prompt
            })
        
        # 如果有图像输入
        if image is not None:
            print("处理输入图像...")
            image_base64 = self.tensor_to_base64(image)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                }
            })
        
        # 如果有视频输入
        if video_frames is not None:
            print("处理视频帧...")
            frame_data, frame_count = self.process_video_frames(video_frames)
            # 将视频帧作为多个图像添加
            for frame_info in frame_data:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": frame_info["image"]
                    }
                })
            # 添加视频说明
            user_content.append({
                "type": "text",
                "text": f"[以上是视频的{frame_count}个关键帧]"
            })
        
        # 添加用户消息到对话历史
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # 准备请求数据
        data = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # 添加思考模式控制
        # 注意：thinking参数需要是对象格式，而不是布尔值
        if thinking_mode == "thinking":
            data["thinking"] = {"enabled": True}
        elif thinking_mode == "non-thinking":
            data["thinking"] = {"enabled": False}
        # auto模式下不设置thinking参数，让模型自动决定
        
        # 如果有seed，添加到请求中
        if seed != -1:
            if seed < 0:
                seed = 0
            elif seed > 2147483647:
                seed = seed % 2147483648
            data["seed"] = seed
        
        # 根据思考模式调整超时时间
        actual_timeout = timeout
        if thinking_mode == "thinking":
            # 思考模式需要更长时间
            actual_timeout = max(timeout, 300)
        
        # 打印调试信息
        print(f"正在调用对话API...")
        print(f"模型: {model_id}")
        print(f"思考模式: {thinking_mode}")
        print(f"温度: {temperature}")
        print(f"最大输出: {max_tokens}")
        print(f"超时时间: {actual_timeout}秒")
        
        # 发送请求
        try:
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=data,
                timeout=actual_timeout
            )
            
            if response.status_code != 200:
                error_msg = f"API调用失败: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" - {json.dumps(error_data['error'], ensure_ascii=False)}"
                    else:
                        error_msg += f" - {response.text}"
                except:
                    error_msg += f" - {response.text}"
                raise RuntimeError(error_msg)
            
            # 解析响应
            result = response.json()
            
            # 提取回复内容
            assistant_message = result.get("choices", [{}])[0].get("message", {})
            response_content = assistant_message.get("content", "")
            
            # 提取思考内容（如果有）
            thinking_content = ""
            thinking_tokens = 0
            if "thinking_content" in assistant_message:
                thinking_content = assistant_message.get("thinking_content", "")
                # 获取思考过程的token数
                usage = result.get("usage", {})
                thinking_tokens = usage.get("thinking_tokens", 0)
            
            # 添加助手回复到对话历史
            messages.append({
                "role": "assistant",
                "content": response_content
            })
            
            # 获取token使用情况
            usage = result.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            
            # 输出信息
            print(f"对话完成！")
            print(f"总token使用: {total_tokens}")
            if thinking_tokens > 0:
                print(f"思考过程token: {thinking_tokens}")
            
            # 返回结果
            return (
                response_content,
                thinking_content,
                json.dumps(messages, ensure_ascii=False, indent=2),
                float(total_tokens),
                thinking_tokens
            )
            
        except requests.exceptions.Timeout:
            error_msg = f"请求超时（{actual_timeout}秒）"
            if thinking_mode == "thinking":
                error_msg += "\n建议：\n1. 尝试使用 'non-thinking' 模式以获得更快响应\n2. 增加超时时间\n3. 简化输入内容"
            else:
                error_msg += "\n建议：\n1. 检查网络连接\n2. 增加超时时间\n3. 简化输入内容"
            raise RuntimeError(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"网络连接失败: {str(e)}\n请检查网络连接和API端点是否正确")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"对话时出错: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """确保每次都重新执行（用于API调用）"""
        return float("NaN")


class DoubaoLLMHistoryNode:
    """
    管理对话历史的辅助节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": ([
                    "create_new",
                    "clear",
                    "load_from_file",
                    "save_to_file"
                ], {
                    "default": "create_new"
                }),
            },
            "optional": {
                "conversation_json": ("STRING", {
                    "default": "[]",
                    "multiline": True
                }),
                "file_path": ("STRING", {
                    "default": "conversation_history.json",
                    "multiline": False
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "初始系统提示词"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("conversation_json",)
    FUNCTION = "manage_history"
    CATEGORY = "✨即梦AI生成"
    DESCRIPTION = "管理对话历史，支持创建、清空、保存和加载"

    def manage_history(self, action, conversation_json="[]", file_path="", system_prompt=""):
        """管理对话历史"""
        
        if action == "create_new":
            # 创建新的对话历史
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            return (json.dumps(messages, ensure_ascii=False, indent=2),)
        
        elif action == "clear":
            # 清空对话历史
            return ("[]",)
        
        elif action == "load_from_file":
            # 从文件加载
            try:
                if not file_path:
                    file_path = "conversation_history.json"
                
                # 构建完整路径
                if not os.path.isabs(file_path):
                    file_path = os.path.join(os.path.dirname(__file__), "conversations", file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return (json.dumps(data, ensure_ascii=False, indent=2),)
            except Exception as e:
                print(f"加载对话历史失败: {str(e)}")
                return ("[]",)
        
        elif action == "save_to_file":
            # 保存到文件
            try:
                if not file_path:
                    file_path = "conversation_history.json"
                
                # 解析JSON
                messages = json.loads(conversation_json)
                
                # 构建完整路径
                if not os.path.isabs(file_path):
                    save_dir = os.path.join(os.path.dirname(__file__), "conversations")
                    os.makedirs(save_dir, exist_ok=True)
                    file_path = os.path.join(save_dir, file_path)
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2)
                
                print(f"对话历史已保存到: {file_path}")
                return (conversation_json,)
            except Exception as e:
                print(f"保存对话历史失败: {str(e)}")
                return (conversation_json,)
        
        return (conversation_json,) 