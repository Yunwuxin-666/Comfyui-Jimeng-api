import os
import torch
import numpy as np
from PIL import Image
import requests
import json
import time
import base64
from io import BytesIO
import tempfile
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
import urllib3

# 导入网络配置
try:
    from network_config import *
except ImportError:
    # 如果配置文件不存在，使用默认配置
    CONNECTION_TIMEOUT = 15
    READ_TIMEOUT = 600
    MAX_RETRIES = 5
    BACKOFF_FACTOR = 2.0
    MAX_WAIT_TIME = 30
    POOL_CONNECTIONS = 20
    POOL_MAXSIZE = 50
    POOL_BLOCK = False
    STATUS_QUERY_INTERVAL = 5
    MAX_STATUS_QUERIES = 60
    STATUS_RETRY_ATTEMPTS = 3
    NETWORK_DIAGNOSIS_TIMEOUT = 5
    NETWORK_CHECK_INTERVAL = 300
    API_ENDPOINTS = {
        "primary": "https://ark.cn-beijing.volces.com",
        "fallback": "http://ark.cn-beijing.volces.com",
        "alternative": "https://ark.cn-shanghai.volces.com"
    }
    PREFER_HTTP = False
    ENABLE_NETWORK_DIAGNOSIS = True
    ENABLE_AUTO_PROTOCOL_SWITCH = True
    ENABLE_CONNECTION_WARMUP = True
    LOG_LEVEL = "INFO"
    SAVE_DIAGNOSIS_TO_FILE = False
    DIAGNOSIS_FILE_PATH = "network_diagnosis.log"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Seedream4StreamingNode:
    """
    Seedream 4.0 流式响应节点
    支持多图输入输出和流式生成体验
    模型：doubao-seedream-4-0-250828
    """
    
    # 预定义的模型ID选项
    MODEL_IDS = {
        "doubao-seedream-4-0-250828": "doubao-seedream-4-0-250828"
    }
    
    def __init__(self):
        # 禁用不安全请求警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 网络诊断状态
        self.network_status = {
            'http_available': True,
            'https_available': True,
            'last_check': None,
            'connection_quality': 'unknown'
        }
        
        # 流式状态
        self.streaming_status = {
            'is_streaming': False,
            'current_step': 0,
            'total_steps': 6,
            'progress': 0.0,
            'status_message': ''
        }
    
    def diagnose_network(self, endpoint_base="ark.cn-beijing.volces.com"):
        """诊断网络连接状态 - HTTPS优先策略"""
        print("🔍 网络诊断中...")
        
        # HTTPS优先检测
        test_endpoints = [
            f"https://{endpoint_base}",
            f"http://{endpoint_base}"
        ]
        
        for endpoint in test_endpoints:
            try:
                session = requests.Session()
                session.timeout = (5, 10)
                
                response = session.get(f"{endpoint}/api/v3/contents/generations/tasks", timeout=(5, 10))
                
                if endpoint.startswith("https://"):
                    self.network_status['https_available'] = True
                else:
                    self.network_status['http_available'] = True
                    
            except Exception as e:
                if endpoint.startswith("https://"):
                    self.network_status['https_available'] = False
                else:
                    self.network_status['http_available'] = False
        
        # 更新连接质量评估
        if self.network_status['https_available']:
            if self.network_status['http_available']:
                self.network_status['connection_quality'] = 'excellent (HTTPS+HTTP)'
            else:
                self.network_status['connection_quality'] = 'good (HTTPS only)'
        elif self.network_status['http_available']:
            self.network_status['connection_quality'] = 'good (HTTP only)'
        else:
            self.network_status['connection_quality'] = 'poor (no connection)'
        
        self.network_status['last_check'] = time.time()
        print(f"🌐 网络状态: {self.network_status['connection_quality']}")
        
        return self.network_status
    
    def create_robust_session(self, max_retries=5, backoff_factor=2.0):
        """创建具有智能重试机制和连接优化的requests会话"""
        session = requests.Session()
        
        # 配置智能重试策略
        try:
            retry_strategy = Retry(
                total=max_retries,
                status_forcelist=[408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
                allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
                backoff_factor=backoff_factor,
                raise_on_redirect=False,
                raise_on_status=False,
                respect_retry_after_header=True
            )
        except TypeError:
            try:
                retry_strategy = Retry(
                    total=max_retries,
                    status_forcelist=[408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
                    method_whitelist=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
                    backoff_factor=backoff_factor,
                    raise_on_redirect=False,
                    raise_on_status=False,
                    respect_retry_after_header=True
                )
            except Exception as e:
                retry_strategy = Retry(
                    total=max_retries,
                    status_forcelist=[500, 502, 503, 504],
                    backoff_factor=backoff_factor,
                    raise_on_redirect=False,
                    raise_on_status=False
                )
        
        # 配置HTTP适配器
        try:
            adapter_kwargs = {
                'max_retries': retry_strategy,
                'pool_connections': 20,
                'pool_maxsize': 50,
                'pool_block': False
            }
            
            try:
                adapter = HTTPAdapter(**adapter_kwargs, pool_connections_retry=3)
            except TypeError:
                adapter = HTTPAdapter(**adapter_kwargs)
                
        except Exception as e:
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=20,
                pool_block=False
            )
        
        # 挂载适配器
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置超时
        session.timeout = (15, 600)
        
        # 设置请求头优化
        session.headers.update({
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate',
            'User-Agent': 'ComfyUI-Seedream4-Node/1.0'
        })
        
        return session
    
    def update_streaming_status(self, step, total_steps, message=""):
        """更新流式状态"""
        self.streaming_status['current_step'] = step
        self.streaming_status['total_steps'] = total_steps
        self.streaming_status['progress'] = (step / total_steps) * 100
        self.streaming_status['status_message'] = message
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        # 显示进度条
        if total_steps > 0:
            bar_length = 20
            filled_length = int(bar_length * self.streaming_status['progress'] / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"进度条: |{bar}| {self.streaming_status['progress']:.1f}%")
    
    def simulate_streaming_progress(self, total_steps, current_step, enable_streaming=True):
        """模拟流式进度显示"""
        if not enable_streaming:
            return
        
        progress = (current_step / total_steps) * 100
        
        status_messages = [
            "🔄 正在分析提示词...",
            "🎯 正在生成图像...", 
            "✨ 正在优化细节...",
            "🎨 正在应用风格...",
            "🔍 正在进行质量检查...",
            "📸 正在生成最终图像..."
        ]
        
        message = status_messages[min(current_step, len(status_messages) - 1)]
        self.update_streaming_status(current_step, total_steps, message)
        
        # 模拟流式延迟
        time.sleep(0.5)
    
    def update_progress(self, current_step, total_steps, message=""):
        """更新进度显示并返回进度值"""
        progress = (current_step + 1) / total_steps * 100
        print(f"📊 进度: {progress:.1f}% - {message}")
        return progress
    
    def tensor_to_base64(self, tensor_image):
        """将张量图像转换为base64字符串 - 参考seededit.py"""
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
    
    def process_multi_images(self, input_images, batch_size):
        """处理多图输入输出"""
        if input_images is not None:
            if len(input_images.shape) == 4:  # 批次维度
                batch_count = input_images.shape[0]
                print(f"📸 检测到 {batch_count} 张输入图像")
                
                # 为每张图像生成对应的输出
                processed_images = []
                for i in range(batch_count):
                    img = input_images[i]
                    processed_img = self.process_single_image(img)
                    processed_images.append(processed_img)
                
                return torch.stack(processed_images)
            else:
                # 单张图像
                return self.process_single_image(input_images)
        else:
            # 纯文本生成
            return self.generate_from_text(batch_size)
    
    def process_single_image(self, image_tensor):
        """处理单张图像"""
        # 这里可以添加图像预处理逻辑
        return image_tensor
    
    def generate_from_text(self, batch_size):
        """从文本生成图像"""
        # 这里实现文本到图像的生成逻辑
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "placeholder": "描述您想要生成的图像内容"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的火山引擎API Key"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "number"
                }),
                "resolution": ([
                    "1K",
                    "2K", 
                    "4K"
                ], {
                    "default": "2K"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "sequential_image_generation": (["disabled", "auto"], {
                    "default": "disabled",
                    "tooltip": "序列生成控制：disabled=多图融合，auto=自动序列生成"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用水印",
                    "label_off": "禁用水印"
                })
            },
            "optional": {
                "image1": ("IMAGE", {
                    "tooltip": "输入图像1"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "输入图像2"
                }),
                "image3": ("IMAGE", {
                    "tooltip": "输入图像3"
                }),
                "image4": ("IMAGE", {
                    "tooltip": "输入图像4"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "task_id", "status", "batch_count", "generation_time", "progress")
    FUNCTION = "generate_images"
    CATEGORY = "✨即梦AI生成"
    DESCRIPTION = """Seedream 4.0 图像生成节点
    
支持特性：
- 多图融合（多图输入单图输出）
- 序列生成（多图输入多图输出）
- 高质量图像生成（1K/2K/4K）
- 水印控制
- 引导强度调节

模型：doubao-seedream-4-0-250828
"""
    
    def generate_images(self, prompt, api_key, batch_size, resolution, guidance_scale, 
                       sequential_image_generation="disabled", watermark=False, 
                       image1=None, image2=None, image3=None, image4=None):
        """生成图像的主函数"""
        
        start_time = time.time()
        
        # 读取API Key
        apikey_path = os.path.join(os.path.dirname(__file__), "apikey.txt")
        if os.path.exists(apikey_path):
            with open(apikey_path, "r", encoding="utf-8") as f:
                file_api_key = f.read().strip()
            api_key = file_api_key if file_api_key else api_key
        
        if not api_key:
            raise ValueError("请提供有效的API Key")
        
        # 固定使用Seedream 4.0模型
        model_id = "doubao-seedream-4-0-250828"
        
        print(f"🚀 开始生成图像...")
        print(f"📝 提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"🤖 模型: {model_id}")
        print(f"📐 分辨率: {resolution}")
        print(f"📦 批次大小: {batch_size}")
        print(f"🎯 引导强度: {guidance_scale}")
        print(f"🔄 序列生成: {sequential_image_generation}")
        print(f"🏷️ 水印: {'开启' if watermark else '关闭'}")
        
        # 初始化进度追踪
        total_steps = 6
        current_progress = 0
        
        # 构建请求数据
        data = {
            "model": model_id,
            "prompt": prompt,
            "size": resolution,
            "n": batch_size,
            "response_format": "url",
            "watermark": watermark
        }
        
        # 添加guidance_scale参数
        if guidance_scale != 7.5:
            data["guidance_scale"] = guidance_scale
        
        try:
            # 创建会话
            session = self.create_robust_session()
            
            # 使用正确的API端点 - 参考官方文档和seededit.py
            api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
            print(f"🌐 请求端点: {api_endpoint}")
            
            # 准备请求头
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 处理多图输入 - 收集所有非空的图像输入
            input_images = []
            if image1 is not None:
                input_images.append(image1)
            if image2 is not None:
                input_images.append(image2)
            if image3 is not None:
                input_images.append(image3)
            if image4 is not None:
                input_images.append(image4)
            
            if input_images:
                batch_count = len(input_images)
                print(f"📸 检测到多图输入，数量: {batch_count}")
                
                # 更新进度 - 步骤1: 分析输入
                current_progress = self.update_progress(0, total_steps, "分析输入图像...")
                
                # 根据sequential_image_generation参数决定处理方式
                if sequential_image_generation == "disabled":
                    # 多图融合模式 - 根据官方文档
                    print(f"🔀 使用多图融合模式")
                    
                    # 更新进度 - 步骤2: 处理图像
                    current_progress = self.update_progress(1, total_steps, "处理输入图像...")
                    
                    # 将所有图像转换为base64数组
                    image_array = []
                    for i, img in enumerate(input_images):
                        base64_image = self.tensor_to_base64(img)
                        image_array.append(base64_image)
                        print(f"🖼️ 处理图像 {i+1}: {img.shape}")
                    
                    # 构建多图融合请求 - 参考官方文档格式
                    data.update({
                        "image": image_array,  # 图像数组
                        "sequential_image_generation": "disabled",  # 禁用序列生成
                        "n": 1  # 融合后生成1张图像
                    })
                    
                    print(f"🔄 多图融合请求已构建，图像数量: {len(image_array)}")
                    
                    # 更新进度 - 步骤3: 准备API请求
                    current_progress = self.update_progress(2, total_steps, "准备API请求...")
                    
                elif sequential_image_generation == "auto":
                    # 序列生成模式
                    print(f"🔄 使用序列生成模式")
                    
                    # 更新进度 - 步骤2: 处理图像
                    current_progress = self.update_progress(1, total_steps, "处理输入图像...")
                    
                    # 为每张图像生成对应的输出
                    processed_images = []
                    for i, img in enumerate(input_images):
                        # 更新进度 - 处理每张图像
                        progress_step = 2 + (i * 2) // batch_count
                        current_progress = self.update_progress(progress_step, total_steps, f"处理第 {i+1} 张图像...")
                        
                        base64_image = self.tensor_to_base64(img)
                        
                        # 构建单图请求
                        img_data = {
                            "model": model_id,
                            "prompt": prompt,
                            "image": base64_image,
                            "size": resolution,
                            "n": 1,
                            "response_format": "url",
                            "watermark": watermark
                        }
                        
                        # 添加guidance_scale参数
                        if guidance_scale != 7.5:
                            img_data["guidance_scale"] = guidance_scale
                        
                        # 发送单图请求
                        try:
                            print(f"🖼️ 处理第 {i+1} 张图像...")
                            img_response = session.post(api_endpoint, headers=headers, json=img_data)
                            
                            if img_response.status_code == 200:
                                img_result = img_response.json()
                                generated_images = self.process_response_images(img_result)
                                if len(generated_images) > 0:
                                    processed_images.append(generated_images[0])
                                else:
                                    processed_images.append(img)
                            else:
                                print(f"❌ 第 {i+1} 张图像处理失败: {img_response.status_code}")
                                processed_images.append(img)
                                
                        except Exception as e:
                            print(f"❌ 第 {i+1} 张图像处理异常: {str(e)}")
                            processed_images.append(img)
                    
                    # 返回序列生成结果
                    if processed_images:
                        final_images = torch.stack(processed_images)
                        generation_time = time.time() - start_time
                        
                        print(f"✅ 序列生成完成！")
                        print(f"⏱️ 生成时间: {generation_time:.2f}秒")
                        print(f"📸 处理图像数量: {len(processed_images)}")
                        
                        # 最终进度 - 100%
                        final_progress = self.update_progress(5, total_steps, "序列生成完成！")
                        return (final_images, "sequential_completed", "completed", len(processed_images), generation_time, final_progress)
                    else:
                        raise RuntimeError("序列生成失败，没有生成任何图像")
            else:
                # 纯文本生成模式
                print(f"📝 使用纯文本生成模式")
                # 更新进度 - 步骤1: 分析输入
                current_progress = self.update_progress(0, total_steps, "分析提示词...")
                # 更新进度 - 步骤2: 准备请求
                current_progress = self.update_progress(1, total_steps, "准备API请求...")
            
            # 更新进度 - 步骤4: 发送API请求
            current_progress = self.update_progress(3, total_steps, "发送API请求...")
            
            response = session.post(
                api_endpoint,
                headers=headers,
                json=data
            )
            
            print(f"📊 响应状态码: {response.status_code}")
            print(f"📋 响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"❌ API请求失败: {response.status_code}")
                print(f"📄 响应内容: {response.text[:500]}")
                raise RuntimeError(f"API请求失败: {response.status_code} - {response.text[:200]}")
            
            # 检查响应内容是否为空
            if not response.text.strip():
                raise RuntimeError("API返回空响应")
            
            # 更新进度 - 步骤5: 处理响应
            current_progress = self.update_progress(4, total_steps, "处理API响应...")
            
            # 尝试解析JSON
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {str(e)}")
                print(f"📄 响应内容: {response.text[:1000]}")
                raise RuntimeError(f"API返回非JSON格式响应: {response.text[:200]}")
            
            # 更新进度 - 步骤6: 生成最终结果
            current_progress = self.update_progress(5, total_steps, "生成最终结果...")
            
            # 处理返回的图像
            images = self.process_response_images(result)
            
            generation_time = time.time() - start_time
            
            print(f"✅ 图像生成完成！")
            print(f"⏱️ 生成时间: {generation_time:.2f}秒")
            print(f"📸 生成图像数量: {len(images)}")
            
            # 最终进度 - 100%
            final_progress = self.update_progress(5, total_steps, "任务完成！")
            
            return (images, result.get("task_id", ""), "completed", batch_size, generation_time, final_progress)
            
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            raise
    
    def download_and_convert_image(self, session, image_url):
        """下载图像并转换为张量 - 参考seededit.py"""
        try:
            print(f"🖼️ 正在下载图像: {image_url}")
            
            response = session.get(image_url, timeout=(10, 300))
            response.raise_for_status()
            
            # 转换为PIL图像
            pil_image = Image.open(BytesIO(response.content))
            
            # 确保是RGB格式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(pil_image)
            
            # 归一化到0-1
            image_np = image_np.astype(np.float32) / 255.0
            
            # 添加batch维度
            image_np = np.expand_dims(image_np, axis=0)
            
            # 转换为torch张量
            image_tensor = torch.from_numpy(image_np)
            
            print(f"✅ 图像下载并转换完成")
            
            return image_tensor
            
        except Exception as e:
            print(f"❌ 下载图像失败: {str(e)}")
            # 创建占位符
            placeholder = torch.zeros((1, 1024, 1024, 3))
            return placeholder
    
    def process_response_images(self, result):
        """处理API返回的图像数据 - 参考seededit.py"""
        images = []
        
        print(f"📊 API响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 根据API响应格式处理图像
        if "data" in result:
            # 标准格式：{"data": [{"url": "..."}, ...]}
            for item in result["data"]:
                if "url" in item:
                    image_url = item["url"]
                    print(f"🖼️ 下载图像: {image_url}")
                    
                    # 创建临时会话用于下载
                    temp_session = self.create_robust_session()
                    try:
                        image_tensor = self.download_and_convert_image(temp_session, image_url)
                        images.append(image_tensor[0])  # 移除batch维度
                    except Exception as e:
                        print(f"❌ 下载图像失败: {str(e)}")
                        # 创建占位符
                        placeholder = torch.zeros((1024, 1024, 3))
                        images.append(placeholder)
                        
        elif "images" in result:
            # 备用格式：{"images": ["base64_data", ...]}
            for img_data in result["images"]:
                try:
                    if img_data.startswith("data:image"):
                        # 处理data URL格式
                        img_data = img_data.split(",")[1]
                    
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(BytesIO(img_bytes))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)
                    images.append(img_tensor)
                    
                except Exception as e:
                    print(f"❌ 处理图像数据失败: {str(e)}")
                    placeholder = torch.zeros((1024, 1024, 3))
                    images.append(placeholder)
        
        if not images:
            # 如果没有图像，创建一个占位符
            print("⚠️ 未找到图像数据，创建占位符")
            placeholder = torch.zeros((1024, 1024, 3))
            images.append(placeholder)
        
        return torch.stack(images)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """确保每次都重新执行"""
        return float("NaN")


# 节点注册
NODE_CLASS_MAPPINGS = {
    "Seedream4StreamingNode": Seedream4StreamingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedream4StreamingNode": "Seedream 4.0 流式生成"
}
