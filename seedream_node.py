import os
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json

class SeedreamImageGeneratorNode:
    """
    调用火山引擎即梦API生成图片的节点
    使用 doubao-seedream-3.0-t2i 模型
    """
    
    # 预定义的模型ID选项
    MODEL_IDS = {
        "doubao-seedream-3.0-t2i (基础模型)": "doubao-seedream-3-0-t2i-250415"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一只可爱的猫咪", 
                    "multiline": True,
                    "dynamicPrompts": True
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的火山引擎API Key"
                }),
                "model_selection": (list(cls.MODEL_IDS.keys()), {
                    "default": "doubao-seedream-3.0-t2i (基础模型)"
                }),
                "size": ([
                    "21:5 (2016x512)",
                    "21:9 (2016x864)",
                    "16:9 (1664x936)", 
                    "3:2 (1584x1056)",
                    "4:3 (1472x1104)",
                    "1:1 (1328x1328)",
                    "3:4 (1104x1472)",
                    "2:3 (1056x1584)",
                    "9:16 (936x1664)",
                    "5:21 (512x2016)",
                    "1:1 (1024x1024)",
                    "1:1 (512x512)"
                ], {
                    "default": "1:1 (1024x1024)"
                }),
                                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "添加水印",
                    "label_off": "无水印"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "✨即梦AI生成"
    DESCRIPTION = """使用火山引擎即梦API生成图片。
    
模型：doubao-seedream-3.0-t2i
需要提供有效的API Key才能使用。

如果遇到模型ID错误，请尝试：
1. 在火山方舟控制台创建推理接入点
2. 参考文档：https://www.volcengine.com/docs/82379/1301161
    
参数说明：
- prompt: 图片生成的提示词
- api_key: 火山引擎API密钥
- model_selection: 模型选择
- size: 生成图片的尺寸（显示为"比例 (宽x高)"格式）
- seed: 随机种子（-1表示随机，有效范围：0-2147483647）
- watermark: 是否在生成的图片中添加水印（默认为false，不添加水印）"""

    def generate_image(self, prompt, api_key, model_selection="doubao-seedream-3.0-t2i (基础模型)", 
                      size="1:1 (1024x1024)", seed=-1, watermark=False):
        """调用火山引擎API生成图片"""
        
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
        
        # 设置默认值（这些参数不再显示在UI中）
        response_format = "url"
        custom_model_id = ""
        api_endpoint = "https://ark.cn-beijing.volces.com/api/v3"
        debug_mode = False
        n = 1  # 默认生成1张图片
        
        if not api_key:
            raise ValueError("请提供有效的API Key")
        
        # 确定使用的模型ID
        model_id = self.MODEL_IDS[model_selection]
        
        # 处理 api_endpoint 参数，确保它是有效的字符串
        if api_endpoint is None or api_endpoint == "" or api_endpoint == False or not isinstance(api_endpoint, str):
            api_endpoint = "https://ark.cn-beijing.volces.com/api/v3"
        
        # 额外的安全检查
        api_endpoint = str(api_endpoint).strip()
        if not api_endpoint or api_endpoint.lower() in ['false', 'none', 'null', '0']:
            api_endpoint = "https://ark.cn-beijing.volces.com/api/v3"
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 处理size参数 - 从 "比例 (宽x高)" 格式中提取尺寸
        actual_size = size
        try:
            if " (" in size and ")" in size:
                # 找到括号的位置
                start = size.find("(")
                end = size.find(")")
                if start != -1 and end != -1 and end > start:
                    # 提取括号中的内容
                    actual_size = size[start+1:end]
                    
            # 验证最终结果是有效的尺寸格式（必须包含x）
            if "x" not in actual_size or len(actual_size.split("x")) != 2:
                # 如果不是有效格式，使用默认值
                if debug_mode:
                    print(f"警告：尺寸格式无效 '{actual_size}'（来自 '{size}'），使用默认值 1024x1024")
                actual_size = "1024x1024"
            else:
                # 验证宽高都是数字
                width, height = actual_size.split("x")
                if not (width.isdigit() and height.isdigit()):
                    if debug_mode:
                        print(f"警告：尺寸数值无效 '{actual_size}'，使用默认值 1024x1024")
                    actual_size = "1024x1024"
                    
        except Exception as e:
            # 如果出现任何错误，使用默认值
            if debug_mode:
                print(f"警告：解析尺寸时出错 '{size}': {e}，使用默认值 1024x1024")
            actual_size = "1024x1024"
        
        # 准备请求数据
        data = {
            "model": model_id,
            "prompt": prompt,
            "size": actual_size,
            "n": n,
            "response_format": response_format,
            "watermark": watermark
        }
        
        # 处理seed参数
        # -1 表示随机，不发送seed参数
        # 其他值需要确保在合理范围内
        if seed != -1:
            # 确保seed在32位整数范围内（0 到 2147483647）
            if seed < 0:
                seed = 0
            elif seed > 2147483647:
                seed = seed % 2147483648  # 使用模运算将大数映射到有效范围
            
            # 只有在seed为有效正整数时才添加到请求中
            if seed >= 0:
                data["seed"] = int(seed)
        
        # 调试模式：打印请求信息
        if debug_mode:
            print("=" * 50)
            print("即梦API调试信息")
            print("=" * 50)
            print(f"所有参数:")
            print(f"  - prompt: {prompt}")
            print(f"  - api_key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***'}")
            print(f"  - model_selection: {model_selection}")
            print(f"  - model_id: {model_id}")
            print(f"  - api_endpoint: {api_endpoint}")

            print(f"  - debug_mode: {debug_mode}")
            print(f"  - seed (原始): {seed}")
            print(f"  - seed (处理后): {'随机' if 'seed' not in data else data['seed']}")
            print(f"  - size (原始): {size}")
            print(f"  - size (处理后): {actual_size}")
            print(f"  - watermark: {watermark}")
            print("=" * 50)
            print(f"API端点: {api_endpoint}/images/generations")
            print(f"模型ID: {model_id}")
            print(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            print("=" * 50)
        
        try:
            # 发送请求
            response = requests.post(
                f"{api_endpoint}/images/generations",
                headers=headers,
                json=data,
                timeout=300  # 300秒超时
            )
            
            # 调试模式：打印响应信息
            if debug_mode:
                print(f"响应状态码: {response.status_code}")
                print(f"响应内容: {response.text[:500]}...")  # 只打印前500字符
                print("=" * 50)
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" - {json.dumps(error_data['error'], ensure_ascii=False)}"
                except:
                    error_msg += f" - {response.text}"
                
                # 如果是模型ID错误，提供更详细的帮助信息
                if "InvalidParameter" in response.text and "model" in response.text:
                    error_msg += "\n\n可能的解决方案：\n"
                    error_msg += "1. 在火山方舟控制台创建推理接入点\n"
                    error_msg += "2. 使用推理接入点ID替代模型ID\n"
                    error_msg += "3. 参考文档：https://www.volcengine.com/docs/82379/1301161"
                
                raise RuntimeError(error_msg)
            
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 调试：打印响应中的图片数量
            if debug_mode:
                print(f"API响应中包含 {len(result.get('data', []))} 个图片数据")
                print(f"请求的图片数量: {n}")
            
            # 处理返回的图片
            images = []
            
            if "data" in result:
                for idx, img_data in enumerate(result["data"]):
                    if debug_mode:
                        print(f"处理第 {idx + 1} 张图片...")
                    
                    if response_format == "url":
                        # 从URL下载图片
                        img_url = img_data.get("url")
                        if img_url:
                            img_response = requests.get(img_url, timeout=300)
                            img = Image.open(BytesIO(img_response.content))
                    else:
                        # 从base64解码图片
                        import base64
                        b64_string = img_data.get("b64_json")
                        if b64_string:
                            img_bytes = base64.b64decode(b64_string)
                            img = Image.open(BytesIO(img_bytes))
                    
                    # 确保图片是RGB格式
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # 转换为tensor
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)
                    images.append(img_tensor)
            
            if not images:
                raise ValueError("API未返回任何图片")
            
            # 将所有图片堆叠成一个batch
            output_images = torch.stack(images, dim=0)
            
            print(f"成功生成 {len(images)} 张图片")
            
            # 注意：ComfyUI的PreviewImage节点默认只显示第一张图片
            # 如需查看所有图片，请使用SaveImage节点或其他支持批量显示的节点
            
            return (output_images,)
            
        except requests.exceptions.RequestException as e:
            if debug_mode:
                print(f"请求异常: {str(e)}")
            raise RuntimeError(f"API请求失败: {str(e)}")
        except Exception as e:
            if debug_mode:
                print(f"生成错误: {str(e)}")
            raise RuntimeError(f"生成图片时出错: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """确保每次都重新执行（用于API调用）"""
        return float("NaN")




 