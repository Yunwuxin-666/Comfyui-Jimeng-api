import os
import torch
import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
import json
import time
import tempfile
import logging
import base64
import ssl
import urllib3

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeedEditImageEditorNode:
    """
    调用火山引擎即梦API进行图像编辑的节点
    支持多种图像编辑功能：局部重绘、超分辨率、风格转换等
    """
    
    def __init__(self):
        # 禁用不安全请求警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def create_robust_session(self, max_retries=3, backoff_factor=1.0):
        """创建具有重试机制和SSL优化的requests会话"""
        session = requests.Session()
        
        # 配置重试策略 - 兼容不同版本的urllib3
        try:
            # 新版本urllib3使用allowed_methods
            retry_strategy = Retry(
                total=max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
                backoff_factor=backoff_factor,
                raise_on_redirect=False,
                raise_on_status=False
            )
        except TypeError:
            # 旧版本urllib3使用method_whitelist
            retry_strategy = Retry(
                total=max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
                backoff_factor=backoff_factor,
                raise_on_redirect=False,
                raise_on_status=False
            )
        
        # 配置HTTP适配器
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
            pool_block=False
        )
        
        # 挂载适配器
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置默认超时
        session.timeout = (10, 300)  # (连接超时, 读取超时)
        
        return session
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入要编辑的图像"
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "placeholder": "描述您想要的编辑效果"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的火山引擎API Key"
                }),

                "strength": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "编辑强度，0为最小，1为最大"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "true",
                    "label_off": "false",
                    "tooltip": "是否在生成的图片中添加水印标识"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "蒙版（仅inpainting模式需要）"
                }),

                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number",
                    "tooltip": "随机种子（-1为随机）"
                }),

            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "task_id", "status")
    FUNCTION = "edit_image"
    CATEGORY = "✨即梦AI生成"
    DESCRIPTION = """使用火山引擎即梦SeedEdit 3.0 API进行图像编辑。
    
模型：doubao-seededit-3-0-i2i-250628

功能说明：
支持通用图像编辑、局部重绘、图像扩展、超分辨率等多种编辑功能。

参数说明：
- image: 输入的原始图像
- prompt: 描述编辑效果的文本
- strength: 编辑强度（0-1）
- watermark: 水印开关，默认关闭（false不添加水印，true添加"AI生成"水印）
- mask: 可选，蒙版（用于局部重绘）
- seed: 随机种子（-1表示随机）"""

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
    
    def mask_to_base64(self, mask_tensor):
        """将蒙版张量转换为base64字符串"""
        # 确保张量在CPU上并转换为numpy
        if len(mask_tensor.shape) == 3:
            mask_tensor = mask_tensor[0]  # 取第一个蒙版
        
        # 转换为numpy数组
        mask_np = mask_tensor.cpu().numpy()
        
        # 如果是归一化的值，转换为0-255
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.uint8)
        
        # 确保是2D数组
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]
        
        # 创建PIL图像（灰度图）
        pil_image = Image.fromarray(mask_np, mode='L')
        
        # 转换为base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    
    def validate_base64_format(self, base64_string):
        """验证Base64格式是否正确"""
        if not base64_string:
            return False, "Base64字符串为空"
        
        # 检查是否包含data URI前缀
        if not base64_string.startswith('data:'):
            return False, "缺少data URI前缀"
        
        # 检查格式：data:image/<format>;base64,<data>
        parts = base64_string.split(',')
        if len(parts) != 2:
            return False, "Base64格式不正确，缺少逗号分隔符"
        
        header = parts[0]
        data = parts[1]
        
        # 检查header格式
        if not header.startswith('data:image/') or ';base64' not in header:
            return False, "Header格式不正确，应为data:image/<format>;base64"
        
        # 检查数据部分
        if not data:
            return False, "Base64数据部分为空"
        
        # 尝试解码验证
        try:
            base64.b64decode(data)
            return True, "格式正确"
        except Exception as e:
            return False, f"Base64解码失败: {str(e)}"
    
    def validate_api_parameters(self, data):
        """验证API参数是否符合火山引擎SeedEdit API要求"""
        # 检查必需参数
        required_params = ["model", "prompt", "image"]
        for param in required_params:
            if param not in data or not data[param]:
                return False, f"缺少必需参数: {param}"
        
        # 验证size参数
        if "size" in data and data["size"] != "adaptive":
            return False, f"size参数值无效: {data['size']}，当前仅支持 'adaptive'"
        
        # 验证response_format参数
        valid_response_formats = ["url", "b64_json"]
        if "response_format" in data and data["response_format"] not in valid_response_formats:
            return False, f"response_format参数值无效: {data['response_format']}，支持: {valid_response_formats}"
        
        # 验证图像格式
        if not data["image"].startswith("data:image/"):
            return False, "图像格式不正确，应为data:image/格式"
        
        return True, "所有参数验证通过"
    
    def base64_to_tensor(self, base64_string):
        """将base64字符串转换为张量图像"""
        # 移除data URL前缀（如果有）
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # 解码base64
        img_data = base64.b64decode(base64_string)
        
        # 转换为PIL图像
        pil_image = Image.open(BytesIO(img_data))
        
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
        
        return image_tensor
    
    def edit_image(self, image, prompt, api_key, strength, watermark=False, 
                   mask=None, seed=-1):
        """调用火山引擎API编辑图像"""
        
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
        
        # 调试：打印所有参数
        logger.info(f"[编辑参数] 提示词: {prompt[:50]}..." if prompt else "[编辑参数] 提示词: 无")
        logger.info(f"[编辑参数] 强度: {strength}")
        logger.info(f"[编辑参数] 水印: {'开启' if watermark else '关闭'}")
        logger.info(f"[编辑参数] 种子: {seed}")
        logger.info(f"[编辑参数] 有蒙版: {'是' if mask is not None else '否'}")
        
        # API配置 - 使用图像编辑API
        api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        
        # 根据火山引擎文档，正确的模型ID
        model_id = "doubao-seededit-3-0-i2i-250628"
        
        # 记录使用的模型
        logger.info(f"[模型选择] 使用模型: {model_id}")
        
        # 构建完整提示词 - 直接使用用户提示词
        full_prompt = prompt if prompt else "优化图像质量，增强细节和色彩"
        
        # 添加参数
        full_prompt += f" --strength {strength}"
        
        # 添加种子（用于可重复生成）
        if seed != -1:
            if seed < 0:
                seed = 0
            elif seed > 2147483647:
                seed = seed % 2147483648
            full_prompt += f" --seed {seed}"
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 处理输入图像
        logger.info("正在处理输入图像...")
        image_base64 = self.tensor_to_base64(image)
        
        # 确保Base64格式正确（包含完整的data URI前缀）
        # 火山引擎API要求格式：data:image/png;base64,<base64_data>
        if not image_base64.startswith('data:'):
            # 如果没有前缀，添加标准前缀
            image_base64 = f"data:image/png;base64,{image_base64}"
        
        # 验证Base64格式
        is_valid, validation_msg = self.validate_base64_format(image_base64)
        if not is_valid:
            raise ValueError(f"图像Base64格式无效: {validation_msg}")
        
        logger.info(f"输入图像处理完成，完整Base64长度: {len(image_base64)}")
        
        # 处理蒙版（如果有）
        mask_base64 = None
        if mask is not None:
            logger.info("正在处理蒙版...")
            mask_base64 = self.mask_to_base64(mask)
            # 确保蒙版Base64格式正确
            if not mask_base64.startswith('data:'):
                mask_base64 = f"data:image/png;base64,{mask_base64}"
            logger.info("蒙版处理完成")
        

        
        # 根据API文档构建请求数据
        # 使用image字段，支持Base64格式（符合官方文档）
        data = {
            "model": model_id,
            "prompt": full_prompt,
            "image": image_base64,  # base64编码的图像数据（包含data:前缀）
            "response_format": "url",  # 返回URL格式，便于下载
            "size": "adaptive",  # 根据火山引擎文档，当前仅支持adaptive
            "watermark": watermark  # 使用用户选择的水印设置
        }
        
        # 如果有蒙版，添加到请求中
        if mask_base64:
            data["mask"] = mask_base64
        

        
        # 添加种子（用于可重复生成）
        if seed != -1:
            if seed < 0:
                seed = 0
            elif seed > 2147483647:
                seed = seed % 2147483648
            data["seed"] = seed
        
        # 验证API参数
        logger.info("正在验证API参数...")
        is_valid, validation_msg = self.validate_api_parameters(data)
        if not is_valid:
            raise ValueError(f"API参数验证失败: {validation_msg}")
        logger.info("API参数验证通过")
        
        # 打印调试信息
        logger.info(f"[API调用] 正在创建图像编辑任务...")
        logger.info(f"[API调用] 端点: {api_endpoint}")
        logger.info(f"[API调用] 模型: {model_id}")
        logger.info(f"[API调用] 提示词: {full_prompt[:100]}...")
        
        # 调试：打印请求数据结构（隐藏图像数据）
        debug_data = {
            "model": data["model"],
            "prompt": data["prompt"][:100] if data.get("prompt") else None,
            "has_image": bool(data.get("image")),
            "image_format": "data:image/png;base64" if data.get("image", "").startswith("data:") else "raw_base64",
            "has_mask": bool(data.get("mask")),
            "size": data.get("size"),
            "response_format": data.get("response_format"),
            "watermark": data.get("watermark")
        }
        logger.info(f"[请求数据] {json.dumps(debug_data, ensure_ascii=False)}")
        
        try:
            # 创建健壮的会话
            session = self.create_robust_session(max_retries=3, backoff_factor=1.0)
            
            # 创建编辑任务
            logger.info(f"[请求详情] POST {api_endpoint}")
            
            # 使用健壮的会话发送请求
            response = session.post(
                api_endpoint,
                headers=headers,
                json=data,
                timeout=(10, 300)  # (连接超时, 读取超时)
            )
            
            # 记录响应详情
            logger.info(f"[响应状态码] {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"创建任务失败: {response.status_code}"
                logger.error(f"[错误响应] 状态码: {response.status_code}")
                
                try:
                    error_data = response.json()
                    logger.error(f"[错误响应体] {json.dumps(error_data, ensure_ascii=False, indent=2)}")
                    
                    # 处理特定的参数错误
                    if error_data.get("code") == "InvalidParameter":
                        param_message = error_data.get("message", "")
                        if "size" in param_message.lower():
                            error_msg += f" - size参数错误: 当前仅支持 'adaptive' 值"
                        elif "image" in param_message.lower():
                            error_msg += f" - image参数错误: 请检查Base64格式是否正确"
                        else:
                            error_msg += f" - 参数错误: {param_message}"
                    elif "error" in error_data:
                        error_msg += f" - {json.dumps(error_data['error'], ensure_ascii=False)}"
                    else:
                        error_msg += f" - {response.text}"
                except Exception as parse_error:
                    logger.error(f"[解析错误响应失败] {str(parse_error)}")
                    error_msg += f" - {response.text}"
                    
                raise RuntimeError(error_msg)
            
            # 解析响应 - 图像编辑API直接返回结果
            result = response.json()
            logger.info(f"[响应结构] {list(result.keys())}")
            
            # 图像编辑API直接返回结果，不需要轮询
            if "data" in result:
                data_list = result.get("data", [])
                if data_list and len(data_list) > 0:
                    image_data = data_list[0]
                    
                    # 优先处理URL格式结果（符合response_format设置）
                    image_url = image_data.get("url")
                    if image_url:
                        logger.info(f"[编辑成功] 图像编辑完成！")
                        logger.info(f"[图像URL] {image_url}")
                        task_id = result.get("id", "url_result")
                        
                        # 下载编辑后的图像
                        logger.info("开始下载编辑后的图像...")
                        edited_image = self.download_and_convert_image(session, image_url)
                        
                        # 返回结果
                        return (edited_image, task_id, "success")
                    
                    # 备用处理base64格式结果
                    b64_json = image_data.get("b64_json")
                    if b64_json:
                        logger.info(f"[编辑成功] 获得base64格式结果")
                        # 直接转换base64为tensor
                        edited_image = self.base64_to_tensor(b64_json)
                        task_id = result.get("id", "base64_result")
                        return (edited_image, task_id, "success")
                    
                    # 如果都没有，抛出错误
                    raise ValueError("响应中未找到图像数据（url或b64_json）")
            else:
                raise ValueError(f"未知的响应格式: {list(result.keys())}")
            
        except (requests.exceptions.SSLError, ssl.SSLError) as e:
            logger.error(f"[SSL连接错误] {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"SSL连接失败，请检查网络连接后重试: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[网络连接错误] {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"网络连接失败，请检查网络设置: {str(e)}")
        except requests.exceptions.Timeout as e:
            logger.error(f"[请求超时错误] {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"请求超时，请检查网络速度后重试: {str(e)}")
        except Exception as e:
            logger.error(f"[编辑图像异常] {type(e).__name__}: {str(e)}")
            logger.error(f"[异常堆栈]", exc_info=True)
            raise RuntimeError(f"编辑图像时出错: {str(e)}")
    
    def download_and_convert_image(self, session, image_url):
        """下载图像并转换为张量"""
        try:
            logger.info(f"[图像下载] 正在下载图像...")
            
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
            
            logger.info(f"[图像下载] 图像下载并转换完成")
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"[下载图像失败] {str(e)}")
            raise
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """确保每次都重新执行（用于API调用）"""
        return float("NaN")


# 注册节点
NODE_CLASS_MAPPINGS = {
    "SeedEditImageEditor": SeedEditImageEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedEditImageEditor": "✨即梦图像编辑",
}
