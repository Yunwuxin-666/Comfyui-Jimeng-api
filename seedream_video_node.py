import os
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json
import time
import tempfile
import cv2
import logging
import base64

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeedreamVideoGeneratorNode:
    """
    调用火山引擎即梦API生成视频的节点
    支持文本到视频、图片到视频和首尾帧控制的视频生成
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "placeholder": "描述您想要生成的视频内容"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的火山引擎API Key"
                }),
                "model_selection": ([
                    "doubao-seedance-1-0-pro-250528",
                    "doubao-seedance-1-0-lite-t2v-250428",
                    "doubao-seedance-1-0-lite-i2v-250428",
                ], {
                    "default": "doubao-seedance-1-0-pro-250528"
                }),
                "duration": ([
                    "5",
                    "10"
                ], {
                    "default": "5"
                }),
                "ratio": ([
                    "21:9",
                    "16:9",
                    "4:3",
                    "1:1",
                    "3:4",
                    "9:16",
                    "9:21",
                    "keep_ratio",
                    "adaptive"
                ], {
                    "default": "adaptive"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "添加水印",
                    "label_off": "无水印"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
                "fps": ([
                    "16",
                    "24"
                ], {
                    "default": "16"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("frames", "frame_count", "fps", "video_url", "task_id")
    FUNCTION = "generate_video"
    CATEGORY = "✨即梦AI生成"
    DESCRIPTION = """使用火山引擎即梦API生成视频。
    
支持模式：
- 文生视频：仅输入文本提示词
- 图生视频：输入首帧图片和文本提示词
- 首尾帧控制：输入首帧和尾帧图片，生成过渡视频
    
模型：
- doubao-seedance-1-0-pro-250528：支持所有模式（文生视频、图生视频、首尾帧控制）
- doubao-seedance-1-0-lite-t2v-250428：仅支持文生视频
- doubao-seedance-1-0-lite-i2v-250428：仅支持图生视频和首尾帧控制
    
参数说明：
- prompt: 视频生成的文本描述
- api_key: 火山引擎API密钥
- model_selection: 选择使用的模型
- duration: 视频时长（5秒或10秒）
- ratio: 视频比例（多种比例可选）
- watermark: 是否添加水印
- image: 可选，首帧图片（用于图生视频）
- end_image: 可选，尾帧图片（用于首尾帧控制）
- seed: 随机种子（-1表示随机）
- fps: 可选，视频帧率（16或24）

生成模式：
- 仅prompt: 文生视频
- prompt + image: 图生视频（首帧控制）
- prompt + image + end_image: 首尾帧控制视频

输出说明：
- frames: 视频帧序列（可直接连接到视频保存节点）
- frame_count: 总帧数
- fps: 帧率
- video_url: 原始视频URL
- task_id: 任务ID"""

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

    def upload_image_to_temp_service(self, image_tensor):
        """上传图片到临时服务并获取URL"""
        try:
            # 首先尝试转换为base64
            base64_image = self.tensor_to_base64(image_tensor)
            
            # 这里可以使用免费的图片托管服务
            # 方案1：使用imgbb
            # 方案2：使用其他临时图片服务
            # 方案3：直接使用base64 URL
            
            # 暂时直接返回base64 URL
            logger.info("使用base64格式的图片数据")
            return base64_image
            
        except Exception as e:
            logger.error(f"处理图片失败: {str(e)}")
            raise

    def download_video(self, video_url, max_retries=3):
        """下载视频文件"""
        for attempt in range(max_retries):
            try:
                logger.info(f"正在下载视频 (尝试 {attempt + 1}/{max_retries})")
                response = requests.get(video_url, stream=True, timeout=60)
                response.raise_for_status()
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    temp_path = tmp_file.name
                
                logger.info(f"视频下载成功: {temp_path}")
                return temp_path
                
            except Exception as e:
                logger.error(f"下载失败 (尝试 {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        
        return None

    def extract_frames(self, video_path):
        """从视频文件提取帧"""
        frames = []
        frame_count = 0
        fps = 30.0  # 默认帧率
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
            
            # 读取所有帧
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # OpenCV使用BGR，转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为float32并归一化到0-1
                frame_float = frame_rgb.astype(np.float32) / 255.0
                
                frames.append(frame_float)
                frame_count += 1
            
            cap.release()
            
            if frames:
                # 转换为torch张量
                # ComfyUI期望的格式是 (batch, height, width, channels)
                frames_array = np.array(frames)
                frames_tensor = torch.from_numpy(frames_array)
                
                logger.info(f"成功提取 {frame_count} 帧")
                return frames_tensor, frame_count, fps
            else:
                raise ValueError("未能从视频中提取任何帧")
                
        except Exception as e:
            logger.error(f"提取帧失败: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except:
                    pass

    def generate_video(self, prompt, api_key, model_selection, duration, ratio, watermark, image=None, end_image=None, seed=-1, fps="16"):
        """调用火山引擎API生成视频"""
        
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
        print(f"[DEBUG] 接收到的参数:")
        print(f"  - prompt: {prompt[:50]}..." if prompt else "  - prompt: None")
        print(f"  - model_selection: {model_selection}")
        print(f"  - duration: {duration}")
        print(f"  - ratio: {ratio}")
        print(f"  - watermark: {watermark}")
        print(f"  - seed: {seed}")
        print(f"  - fps: {fps}")
        print(f"  - image: {'有' if image is not None else '无'}")
        print(f"  - end_image: {'有' if end_image is not None else '无'}")
        
        if not api_key:
            raise ValueError("请提供有效的API Key")
        
        if not prompt and image is None:
            raise ValueError("请提供视频描述文本或输入图片")
        
        # 判断生成模式
        has_start_image = image is not None
        has_end_image = end_image is not None
        
        if has_start_image and has_end_image:
            generation_mode = "首尾帧控制"
        elif has_start_image:
            generation_mode = "图生视频"
        else:
            generation_mode = "文生视频"
        
        print(f"生成模式: {generation_mode}")
        
        # 验证模型是否支持当前模式
        if model_selection == "doubao-seedance-1-0-lite-t2v-250428":
            # t2v模型仅支持文生视频
            if generation_mode != "文生视频":
                raise ValueError(f"模型 {model_selection} 仅支持文生视频模式，当前模式为：{generation_mode}")
        elif model_selection == "doubao-seedance-1-0-lite-i2v-250428":
            # i2v模型不支持纯文生视频
            if generation_mode == "文生视频":
                raise ValueError(f"模型 {model_selection} 不支持纯文生视频，请提供至少一张输入图片")
        # pro模型支持所有模式，无需验证
        
        # API配置
        api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
        model_id = model_selection  # 使用用户选择的模型
        
        # 构建提示词（添加参数）
        full_prompt = prompt if prompt else ""
        full_prompt += f" --ratio {ratio} --dur {duration}"
        
        # 添加水印控制
        if not watermark:
            full_prompt += " --no-watermark"
        
        # 如果有seed，添加到提示词中
        if seed != -1:
            if seed < 0:
                seed = 0
            elif seed > 2147483647:
                seed = seed % 2147483648
            full_prompt += f" --seed {seed}"
        
        # 如果有fps，添加到提示词中
        if fps:
            full_prompt += f" --fps {fps}"
        
        # 准备请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 准备内容数组
        content = []
        
        # 添加文本内容（文本必须放在第一个）
        if full_prompt:
            content.append({
                "type": "text",
                "text": full_prompt
            })
        
        # 如果有首帧图片，添加图片内容
        if has_start_image:
            print("正在处理首帧图片...")
            # 获取图片URL（base64或上传后的URL）
            start_image_url = self.upload_image_to_temp_service(image)
            content.append({
                "type": "image_url",
                "role": "first_frame",
                "image_url": {
                    "url": start_image_url
                }
            })
            print("首帧图片处理完成")
        
        # 如果有尾帧图片，添加图片内容
        if has_end_image:
            print("正在处理尾帧图片...")
            # 获取图片URL（base64或上传后的URL）
            end_image_url = self.upload_image_to_temp_service(end_image)
            content.append({
                "type": "image_url",
                "role": "last_frame",
                "image_url": {
                    "url": end_image_url
                }
            })
            print("尾帧图片处理完成")
        
        # 准备请求数据
        data = {
            "model": model_id,
            "content": content
        }
        
        # 打印调试信息
        print(f"正在创建{generation_mode}任务...")
        print(f"模型: {model_id}")
        print(f"水印: {'开启' if watermark else '关闭'}")
        if fps:
            print(f"帧率: {fps}fps")
        print(f"提示词: {full_prompt}")
        print(f"内容数量: {len(content)} 项")
        
        try:
            # 创建视频生成任务
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=data,
                timeout=300
            )
            
            if response.status_code != 200:
                error_msg = f"创建任务失败: {response.status_code}"
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
            task_id = result.get("id")
            
            if not task_id:
                raise ValueError("未能获取任务ID")
            
            print(f"{generation_mode}任务已创建，任务ID: {task_id}")
            print(f"任务状态: {result.get('status', 'unknown')}")
            
            # 轮询查询任务状态
            query_url = f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"
            max_attempts = 60  # 最多等待5分钟（5秒一次）
            attempts = 0
            video_url = None
            final_status = "processing"
            
            while attempts < max_attempts:
                time.sleep(5)  # 每5秒查询一次
                attempts += 1
                
                # 查询任务状态
                status_response = requests.get(
                    query_url,
                    headers=headers,
                    timeout=60
                )
                
                if status_response.status_code != 200:
                    print(f"查询任务状态失败: {status_response.status_code}")
                    continue
                
                status_data = status_response.json()
                status = status_data.get("status")
                final_status = status
                
                print(f"任务状态: {status} (尝试 {attempts}/{max_attempts})")
                
                if status == "succeeded":
                    # 获取视频URL
                    content = status_data.get("content", {})
                    video_url = content.get("video_url")
                    
                    if video_url:
                        print(f"{generation_mode}成功！")
                        print(f"视频URL: {video_url}")
                        
                        # 获取使用信息
                        usage = status_data.get("usage", {})
                        if usage:
                            print(f"Token使用: {usage.get('total_tokens', 'N/A')}")
                        break
                    else:
                        raise ValueError("生成成功但未找到视频URL")
                    
                elif status == "failed":
                    error_msg = status_data.get("error", {}).get("message", "未知错误")
                    raise RuntimeError(f"{generation_mode}失败: {error_msg}")
                    
                elif status == "canceled":
                    raise RuntimeError(f"{generation_mode}任务已被取消")
            
            if not video_url:
                raise RuntimeError(f"{generation_mode}超时，最终状态: {final_status}")
            
            # 下载视频并提取帧
            logger.info("开始下载和处理视频...")
            video_path = self.download_video(video_url)
            frames, frame_count, fps = self.extract_frames(video_path)
            
            # 返回结果
            return (frames, frame_count, fps, video_url, task_id)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"生成视频时出错: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """确保每次都重新执行（用于API调用）"""
        return float("NaN")


class SeedreamVideoPreviewNode:
    """
    预览即梦生成的视频
    支持从URL下载或从帧序列预览
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "frames": ("IMAGE",),
                "video_url": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "save_video": ("BOOLEAN", {
                    "default": True,
                    "label_on": "保存到本地",
                    "label_off": "仅预览"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("local_path", "frames")
    FUNCTION = "preview_video"
    CATEGORY = "✨即梦AI生成"
    OUTPUT_NODE = True
    
    def preview_video(self, frames=None, video_url="", save_video=True):
        """预览和保存视频"""
        
        local_path = ""
        output_frames = frames
        
        # 如果有帧数据，直接使用
        if frames is not None:
            if save_video:
                # 创建输出目录
                output_dir = os.path.join(os.path.dirname(__file__), "output", "videos")
                os.makedirs(output_dir, exist_ok=True)
                
                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"seedream_video_{timestamp}_frames.mp4"
                local_path = os.path.join(output_dir, filename)
                
                print(f"视频帧已准备，保存路径: {local_path}")
            
        # 如果只有URL，尝试下载
        elif video_url:
            if save_video:
                try:
                    # 下载视频
                    print(f"正在下载视频: {video_url}")
                    response = requests.get(video_url, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    # 创建输出目录
                    output_dir = os.path.join(os.path.dirname(__file__), "output", "videos")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 生成文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"seedream_video_{timestamp}.mp4"
                    local_path = os.path.join(output_dir, filename)
                    
                    # 保存视频
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"视频已保存到: {local_path}")
                    
                except Exception as e:
                    print(f"下载视频失败: {str(e)}")
                    local_path = video_url  # 如果下载失败，返回原始URL
            else:
                local_path = video_url
        else:
            raise ValueError("请提供视频帧数据或视频URL")
        
        # 返回结果
        return {
            "ui": {
                "videos": [{
                    "url": video_url if video_url else "",
                    "local_path": local_path if save_video else ""
                }]
            },
            "result": (local_path, output_frames)
        } 