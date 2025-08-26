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
import cv2
import logging
import base64
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
    # POOL_CONNECTIONS_RETRY = 3  # 已移除，兼容性问题
    STATUS_QUERY_INTERVAL = 5
    MAX_STATUS_QUERIES = 60
    STATUS_RETRY_ATTEMPTS = 3
    NETWORK_DIAGNOSIS_TIMEOUT = 5
    NETWORK_CHECK_INTERVAL = 300
    API_ENDPOINTS = {
        "primary": "https://ai-budxed1rqdd15m1oi.speedifyvolcai.com",    # 优先使用HTTPS
        "fallback": "http://ai-budxed1rqdd15m1oi.speedifyvolcai.com",    # 备用使用HTTP
        "alternative": "https://ark.cn-shanghai.volces.com" # 备用区域端点
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

class SeedreamVideoGeneratorNode:
    """
    调用火山引擎即梦API生成视频的节点
    支持文本到视频、图片到视频和首尾帧控制的视频生成
    """
    
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
    
    def diagnose_network(self, endpoint_base="ai-budxed1rqdd15m1oi.speedifyvolcai.com"):
        """诊断网络连接状态 - HTTPS优先策略"""
        print("🔍 网络诊断中...")
        
        # HTTPS优先检测
        test_endpoints = [
            f"https://{endpoint_base}",  # 优先检测HTTPS
            f"http://{endpoint_base}"    # 备用检测HTTP
        ]
        
        for endpoint in test_endpoints:
            try:
                session = requests.Session()
                session.timeout = (5, 10)  # 快速测试
                
                # 记录诊断开始时间
                diagnosis_start_time = time.time()
                
                # 解析域名获取IP地址
                try:
                    import socket
                    domain = endpoint_base
                    ip_addresses = socket.gethostbyname_ex(domain)
                    primary_ip = ip_addresses[2][0] if ip_addresses[2] else "未知"
                    all_ips = ", ".join(ip_addresses[2]) if ip_addresses[2] else "未知"
                    print(f"🌐 DNS解析: {domain} → {all_ips}")
                except Exception as dns_error:
                    primary_ip = "DNS解析失败"
                    all_ips = "DNS解析失败"
                    print(f"⚠️ DNS解析失败: {domain} - {str(dns_error)}")
                
                response = session.get(f"{endpoint}/api/v3/contents/generations/tasks", timeout=(5, 10))
                
                # 计算诊断耗时
                diagnosis_duration = time.time() - diagnosis_start_time
                protocol = "HTTPS" if endpoint.startswith("https://") else "HTTP"
                print(f"✅ {endpoint} 连接正常 (状态码: {response.status_code}, 耗时: {diagnosis_duration:.2f}秒, IP: {primary_ip})")
                
                # 打印响应Header信息
                print(f"📋 [{protocol}] 诊断响应Header:")
                for key, value in response.headers.items():
                    print(f"   {key}: {value}")
                
                if endpoint.startswith("https://"):
                    self.network_status['https_available'] = True
                else:
                    self.network_status['http_available'] = True
                    
            except Exception as e:
                # 计算诊断耗时（即使失败也要记录）
                diagnosis_duration = time.time() - diagnosis_start_time
                protocol = "HTTPS" if endpoint.startswith("https://") else "HTTP"
                print(f"❌ {endpoint} 连接失败: {type(e).__name__} (耗时: {diagnosis_duration:.2f}秒, IP: {primary_ip if 'primary_ip' in locals() else '未知'})")
                if endpoint.startswith("https://"):
                    self.network_status['https_available'] = False
                else:
                    self.network_status['http_available'] = False
        
        # 更新连接质量评估 - HTTPS优先
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
        print(f"🌐 网络质量: {self.network_status['connection_quality']}")
        
        return self.network_status
    
    def create_robust_session(self, max_retries=5, backoff_factor=2.0, use_https_fallback=True):
        """创建具有智能重试机制和连接优化的requests会话"""
        session = requests.Session()
        
        # 配置智能重试策略 - 兼容不同版本的urllib3
        try:
            # 新版本urllib3使用allowed_methods
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
            # 旧版本urllib3使用method_whitelist
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
                # 使用最基本的重试配置
                retry_strategy = Retry(
                    total=max_retries,
                    status_forcelist=[500, 502, 503, 504],
                    backoff_factor=backoff_factor,
                    raise_on_redirect=False,
                    raise_on_status=False
                )
        
        # 配置HTTP适配器 - 优化连接池（兼容性处理）
        try:
            import urllib3
            
            # 根据版本选择参数
            adapter_kwargs = {
                'max_retries': retry_strategy,
                'pool_connections': 20,  # 增加连接池大小
                'pool_maxsize': 50,      # 增加最大连接数
                'pool_block': False
            }
            
            # 尝试使用新版本参数
            try:
                adapter = HTTPAdapter(**adapter_kwargs, pool_connections_retry=3)
            except TypeError:
                # 降级到标准配置
                adapter = HTTPAdapter(**adapter_kwargs)
                
        except Exception as e:
            # 使用最基本的配置
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=20,
                pool_block=False
            )
        
        # 挂载适配器
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置更合理的超时 - 连接超时短，读取超时长
        session.timeout = (15, 600)  # (连接超时15秒, 读取超时10分钟)
        
        # 设置请求头优化
        session.headers.update({
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate',
            'User-Agent': 'ComfyUI-Seedream-Node/1.0'
        })
        
        # 连接池健康检查
        try:
            # 测试连接池是否正常工作
            health_check_start_time = time.time()
            
            # 解析健康检查域名
            try:
                import socket
                health_domain = "httpbin.org"
                health_ips = socket.gethostbyname_ex(health_domain)
                health_ip = health_ips[2][0] if health_ips[2] else "未知"
                print(f"🌐 健康检查DNS解析: {health_domain} → {health_ip}")
            except Exception as dns_error:
                health_ip = "DNS解析失败"
                print(f"⚠️ 健康检查DNS解析失败: {health_domain} - {str(dns_error)}")
            
            test_response = session.get("http://httpbin.org/get", timeout=(5, 10))
            health_check_duration = time.time() - health_check_start_time
            
            if test_response.status_code == 200:
                print(f"✅ 连接池配置正常 (健康检查耗时: {health_check_duration:.2f}秒, IP: {health_ip})")
                # 打印健康检查响应Header信息
                print(f"📋 健康检查响应Header:")
                for key, value in test_response.headers.items():
                    print(f"   {key}: {value}")
            else:
                print(f"⚠️ 连接池健康检查异常: {test_response.status_code} (耗时: {health_check_duration:.2f}秒, IP: {health_ip})")
        except Exception as e:
            health_check_duration = time.time() - health_check_start_time
            print(f"⚠️ 连接池健康检查失败，继续使用当前配置 (耗时: {health_check_duration:.2f}秒, IP: {health_ip if 'health_ip' in locals() else '未知'})")
        
        return session
    
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
                    "label_on": "true",
                    "label_off": "false",
                    "tooltip": "是否在生成的图片中添加水印标识" 
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
        session = self.create_robust_session(max_retries=max_retries)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[视频下载] 正在下载视频 (尝试 {attempt + 1}/{max_retries})")
                logger.info(f"[下载URL] {video_url}")
                
                # 使用健壮的会话下载
                response = session.get(video_url, stream=True, timeout=(10, 300))
                
                logger.info(f"[下载响应] 状态码: {response.status_code}")
                logger.info(f"[内容类型] {response.headers.get('Content-Type', 'Unknown')}")
                logger.info(f"[内容长度] {response.headers.get('Content-Length', 'Unknown')} bytes")
                
                response.raise_for_status()
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                            downloaded_size += len(chunk)
                    temp_path = tmp_file.name
                
                logger.info(f"[下载完成] 视频已保存到: {temp_path}")
                logger.info(f"[文件大小] {downloaded_size} bytes")
                return temp_path
                
            except (requests.exceptions.SSLError, ssl.SSLError) as e:
                logger.error(f"[SSL错误] 尝试 {attempt + 1}: {str(e)}")
                logger.error(f"[URL] {video_url}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.info(f"[SSL重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.HTTPError as e:
                logger.error(f"[HTTP错误] 尝试 {attempt + 1}: {str(e)}")
                logger.error(f"[响应状态] {e.response.status_code if e.response else 'No response'}")
                logger.error(f"[响应内容] {e.response.text[:500] if e.response else 'No response'}")
                if attempt == max_retries - 1:
                    raise
                    
            except requests.exceptions.ConnectionError as e:
                logger.error(f"[连接错误] 尝试 {attempt + 1}: {str(e)}")
                logger.error(f"[URL] {video_url}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.info(f"[连接重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise
                    
            except requests.exceptions.Timeout as e:
                logger.error(f"[超时错误] 尝试 {attempt + 1}: {str(e)}")
                logger.error(f"[URL] {video_url}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.info(f"[超时重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"[未知错误] 尝试 {attempt + 1}: {type(e).__name__}: {str(e)}")
                logger.error(f"[URL] {video_url}")
                if attempt == max_retries - 1:
                    raise
                else:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.info(f"[通用重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
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
        
        # 调试：打印关键参数
        print(f"📋 参数检查:")
        print(f"  - 模型: {model_selection}")
        print(f"  - 时长: {duration}秒")
        print(f"  - 比例: {ratio}")
        print(f"  - 水印: {'开启' if watermark else '关闭'}")
        print(f"  - 种子: {seed}")
        print(f"  - 帧率: {fps}fps")
        print(f"  - 首帧: {'有' if image is not None else '无'}")
        print(f"  - 尾帧: {'有' if end_image is not None else '无'}")
        
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
        
        # 网络诊断 - 检查连接状态
        try:
            network_status = self.diagnose_network()
            quality = network_status['connection_quality']
            print(f"🌐 网络状态: {quality}")
            
            # HTTPS优先策略 - 根据网络状态选择最优协议
            if network_status['https_available']:
                # HTTPS可用，优先使用
                api_endpoint = "https://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3/contents/generations/tasks"
                print(f"🔒 使用HTTPS协议 (优先)")
            elif network_status['http_available']:
                # 只有HTTP可用，使用HTTP
                api_endpoint = "http://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3/contents/generations/tasks"
                print(f"🌐 使用HTTP协议 (备用)")
            else:
                # 两种协议都不可用，默认使用HTTPS
                api_endpoint = "https://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3/contents/generations/tasks"
                print(f"⚠️ 网络连接异常，默认使用HTTPS")
                
        except Exception as e:
            logger.warning(f"网络诊断失败: {str(e)}，使用HTTPS默认配置")
            api_endpoint = "https://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3/contents/generations/tasks"
        
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
        
        # API配置 - HTTPS优先，支持HTTP降级
        # 注意：api_endpoint已在网络诊断中设置，这里不需要重复设置
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
            print("🖼️ 处理首帧图片...")
            # 获取图片URL（base64或上传后的URL）
            start_image_url = self.upload_image_to_temp_service(image)
            content.append({
                "type": "image_url",
                "role": "first_frame",
                "image_url": {
                    "url": start_image_url
                }
            })
            print("✅ 首帧图片处理完成")
        
        # 如果有尾帧图片，添加图片内容
        if has_end_image:
            print("🖼️ 处理尾帧图片...")
            # 获取图片URL（base64或上传后的URL）
            end_image_url = self.upload_image_to_temp_service(end_image)
            content.append({
                "type": "image_url",
                "role": "last_frame",
                "image_url": {
                    "url": end_image_url
                }
            })
            print("✅ 尾帧图片处理完成")
        
        # 准备请求数据
        data = {
            "model": model_id,
            "content": content
        }
        
        # 打印任务信息
        print(f"🎬 创建{generation_mode}任务")
        print(f"🤖 模型: {model_id}")
        print(f"💧 水印: {'开启' if watermark else '关闭'}")
        if fps:
            print(f"🎯 帧率: {fps}fps")
        print(f"📝 提示词: {full_prompt[:100]}{'...' if len(full_prompt) > 100 else ''}")
        print(f"📊 内容数量: {len(content)} 项")
        
        try:
            # 智能重试策略 - 支持HTTP降级
            max_attempts = 3
            current_attempt = 0
            response = None
            
            while current_attempt < max_attempts:
                current_attempt += 1
                current_endpoint = api_endpoint
                
                try:
                    # 创建健壮的会话
                    session = self.create_robust_session(max_retries=5, backoff_factor=2.0)
                    
                    # 记录当前尝试和协议信息
                    if current_attempt > 1:
                        print(f"🔄 重试第 {current_attempt} 次")
                    
                    protocol = "HTTPS" if current_endpoint.startswith("https://") else "HTTP"
                    print(f"🌐 使用{protocol}协议请求: {current_endpoint}")
                    
                    # 解析域名获取IP地址
                    try:
                        import socket
                        domain = "ai-budxed1rqdd15m1oi.speedifyvolcai.com"
                        ip_addresses = socket.gethostbyname_ex(domain)
                        primary_ip = ip_addresses[2][0] if ip_addresses[2] else "未知"
                        all_ips = ", ".join(ip_addresses[2]) if ip_addresses[2] else "未知"
                        print(f"🌐 DNS解析: {domain} → {all_ips}")
                    except Exception as dns_error:
                        primary_ip = "DNS解析失败"
                        all_ips = "DNS解析失败"
                        print(f"⚠️ DNS解析失败: {domain} - {str(dns_error)}")
                    
                    # 记录请求开始时间
                    request_start_time = time.time()
                    
                    # 使用健壮的会话发送请求
                    response = session.post(
                        current_endpoint,
                        headers=headers,
                        json=data,
                        timeout=(15, 600)  # (连接超时, 读取超时)
                    )
                    
                    # 计算请求耗时
                    request_duration = time.time() - request_start_time
                    print(f"⏱️ {protocol}请求完成，耗时: {request_duration:.2f}秒, IP: {primary_ip}")
                    
                    # 打印响应Header信息
                    print(f"📋 [{protocol}] 主请求响应Header:")
                    for key, value in response.headers.items():
                        print(f"   {key}: {value}")
                    
                    # 如果成功，跳出重试循环
                    break
                    
                except (requests.exceptions.SSLError, ssl.SSLError) as e:
                    # SSL错误，尝试降级到HTTP
                    if current_endpoint.startswith("https://") and "ai-budxed1rqdd15m1oi.speedifyvolcai.com" in current_endpoint:
                        http_endpoint = current_endpoint.replace("https://", "http://")
                        print(f"🔒 SSL连接失败，降级到HTTP协议")
                        print(f"🔄 切换端点: {current_endpoint} → {http_endpoint}")
                        current_endpoint = http_endpoint
                        api_endpoint = http_endpoint  # 更新全局端点
                        
                        if current_attempt < max_attempts:
                            print(f"⏳ 等待 2 秒后使用HTTP重试...")
                            time.sleep(2)
                            continue
                        else:
                            raise RuntimeError(f"HTTPS和HTTP都连接失败: {str(e)}")
                    else:
                        raise
                        
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if current_attempt < max_attempts:
                        # 计算等待时间 - 指数退避
                        wait_time = min(2 ** current_attempt, 30)  # 最大等待30秒
                        print(f"⏳ 等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        
                        # 如果是HTTPS连接失败，尝试切换到HTTP
                        if current_endpoint.startswith("https://") and "ai-budxed1rqdd15m1oi.speedifyvolcai.com" in current_endpoint:
                            http_endpoint = current_endpoint.replace("https://", "http://")
                            print(f"🔄 连接失败，切换到HTTP协议")
                            print(f"🔄 切换端点: {current_endpoint} → {http_endpoint}")
                            current_endpoint = http_endpoint
                            api_endpoint = http_endpoint  # 更新全局端点
                        continue
                    else:
                        # 所有重试都失败了
                        raise
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"创建任务失败: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" - {json.dumps(error_data['error'], ensure_ascii=False)}"
                    else:
                        error_msg += f" - {response.text}"
                except Exception:
                    error_msg += f" - {response.text}"
                raise RuntimeError(error_msg)
            
            # 解析响应
            result = response.json()
            task_id = result.get("id")
            
            if not task_id:
                raise ValueError("未能获取任务ID")
            
            print(f"✅ 任务创建成功")
            print(f"🆔 任务ID: {task_id}")
            print(f"📊 状态: {result.get('status', 'unknown')}")
            
            # 轮询查询任务状态 - 支持协议降级
            # 根据主请求的协议选择查询协议
            if api_endpoint.startswith("https://"):
                query_url = f"https://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3/contents/generations/tasks/{task_id}"
            else:
                query_url = f"http://ai-budxed1rqdd15m1oi.speedifyvolcai.com/api/v3/contents/generations/tasks/{task_id}"
            
            max_attempts = 60  # 最多等待5分钟（5秒一次）
            attempts = 0
            video_url = None
            final_status = "processing"
            
            print(f"⏳ 开始轮询任务状态...")
            print(f"🔍 查询端点: {query_url}")
            
            while attempts < max_attempts:
                time.sleep(5)  # 每5秒查询一次
                attempts += 1
                
                # 智能重试状态查询
                status_response = None
                status_attempts = 0
                max_status_attempts = 3
                
                while status_attempts < max_status_attempts and status_response is None:
                    status_attempts += 1
                    current_query_url = query_url
                    
                    try:
                        # 查询任务状态
                        protocol = "HTTPS" if current_query_url.startswith("https://") else "HTTP"
                        print(f"🔍 [{protocol}] 查询任务状态: {current_query_url}")
                        
                        # 解析域名获取IP地址
                        try:
                            import socket
                            domain = "ai-budxed1rqdd15m1oi.speedifyvolcai.com"
                            ip_addresses = socket.gethostbyname_ex(domain)
                            primary_ip = ip_addresses[2][0] if ip_addresses[2] else "未知"
                            all_ips = ", ".join(ip_addresses[2]) if ip_addresses[2] else "未知"
                            print(f"🌐 [{protocol}] DNS解析: {domain} → {all_ips}")
                        except Exception as dns_error:
                            primary_ip = "DNS解析失败"
                            all_ips = "DNS解析失败"
                            print(f"⚠️ [{protocol}] DNS解析失败: {domain} - {str(dns_error)}")
                        
                        # 记录查询开始时间
                        query_start_time = time.time()
                        
                        status_response = session.get(
                            current_query_url,
                            headers=headers,
                            timeout=(15, 300)  # (连接超时, 读取超时)
                        )
                        
                        # 计算查询耗时
                        query_duration = time.time() - query_start_time
                        print(f"⏱️ [{protocol}] 状态查询完成，耗时: {query_duration:.2f}秒, IP: {primary_ip}")
                        
                        # 打印响应Header信息
                        print(f"📋 [{protocol}] 状态查询响应Header:")
                        for key, value in status_response.headers.items():
                            print(f"   {key}: {value}")
                        
                        # 如果成功，跳出重试循环
                        break
                        
                    except (requests.exceptions.SSLError, ssl.SSLError) as e:
                        # SSL错误，尝试降级到HTTP
                        if current_query_url.startswith("https://") and "ai-budxed1rqdd15m1oi.speedifyvolcai.com" in current_query_url:
                            http_query_url = current_query_url.replace("https://", "http://")
                            print(f"🔒 SSL查询失败，降级到HTTP协议")
                            print(f"🔄 切换查询端点: {current_query_url} → {http_query_url}")
                            current_query_url = http_query_url
                            query_url = http_query_url  # 更新全局查询URL
                            
                            if status_attempts < max_status_attempts:
                                print(f"⏳ 等待 1 秒后使用HTTP重试...")
                                time.sleep(1)
                                continue
                            else:
                                print(f"⚠️ 状态查询协议降级失败，继续主循环")
                                break
                        else:
                            print(f"❌ 状态查询SSL错误: {str(e)}")
                            break
                        
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        if status_attempts < max_status_attempts:
                            # 计算等待时间 - 指数退避
                            wait_time = min(2 ** status_attempts, 10)  # 最大等待10秒
                            time.sleep(wait_time)
                            
                            # 如果是HTTPS连接失败，尝试切换到HTTP
                            if current_query_url.startswith("https://") and "ai-budxed1rqdd15m1oi.speedifyvolcai.com" in current_query_url:
                                http_query_url = current_query_url.replace("https://", "http://")
                                print(f"🔄 查询连接失败，切换到HTTP协议")
                                print(f"🔄 切换查询端点: {current_query_url} → {http_query_url}")
                                current_query_url = http_query_url
                                query_url = http_query_url  # 更新全局查询URL
                            continue
                        else:
                            print(f"⚠️ 状态查询重试失败，继续主循环")
                            break
                
                if status_response is None:
                    print(f"❌ 状态查询失败，跳过本次查询")
                    continue
                
                if status_response.status_code != 200:
                    print(f"❌ 查询状态失败: {status_response.status_code}")
                    continue
                
                status_data = status_response.json()
                status = status_data.get("status")
                final_status = status
                
                print(f"📊 状态: {status} ({attempts}/{max_attempts})")
                
                if status == "succeeded":
                    # 获取视频URL
                    content = status_data.get("content", {})
                    video_url = content.get("video_url")
                    
                    if video_url:
                        print(f"🎉 {generation_mode}成功！")
                        print(f"🔗 视频URL: {video_url}")
                        
                        # 获取使用信息
                        usage = status_data.get("usage", {})
                        if usage:
                            print(f"💳 Token使用: {usage.get('total_tokens', 'N/A')}")
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
            print(f"📥 开始下载视频...")
            video_path = self.download_video(video_url)
            print(f"🎬 提取视频帧...")
            frames, frame_count, fps = self.extract_frames(video_path)
            
            # 返回结果
            return (frames, frame_count, fps, video_url, task_id)
            
        except (requests.exceptions.SSLError, ssl.SSLError) as e:
            logger.error(f"[SSL连接错误] {type(e).__name__}: {str(e)}")
            logger.error(f"[SSL错误详情] 这通常是由网络不稳定或SSL握手失败造成的")
            logger.error(f"[建议] 系统已自动尝试HTTP降级，如果仍然失败请检查网络连接")
            raise RuntimeError(f"HTTPS和HTTP都连接失败，请检查网络连接后重试: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[网络连接错误] {type(e).__name__}: {str(e)}")
            logger.error(f"[连接错误详情] 无法建立到服务器的连接")
            logger.error(f"[建议] 请检查网络连接和防火墙设置")
            raise RuntimeError(f"网络连接失败，请检查网络设置: {str(e)}")
        except requests.exceptions.Timeout as e:
            logger.error(f"[请求超时错误] {type(e).__name__}: {str(e)}")
            logger.error(f"[超时详情] 请求时间过长导致超时")
            logger.error(f"[建议] 请检查网络速度，稍后重试")
            raise RuntimeError(f"请求超时，请检查网络速度后重试: {str(e)}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"[HTTP错误] {type(e).__name__}: {str(e)}")
            logger.error(f"[HTTP状态码] {e.response.status_code if e.response else 'Unknown'}")
            logger.error(f"[响应内容] {e.response.text[:500] if e.response else 'No response'}")
            raise RuntimeError(f"HTTP请求失败: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[网络请求异常] {type(e).__name__}: {str(e)}")
            logger.error(f"[请求详情] 最后请求的URL: {api_endpoint if 'api_endpoint' in locals() else 'Unknown'}")
            raise RuntimeError(f"网络请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"[生成视频异常] {type(e).__name__}: {str(e)}")
            logger.error(f"[异常堆栈]", exc_info=True)
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
                    # 创建健壮的会话
                    session = self.create_robust_session(max_retries=3)
                    
                    # 下载视频
                    logger.info(f"[预览下载] 正在下载视频: {video_url}")
                    response = session.get(video_url, stream=True, timeout=(10, 300))
                    
                    logger.info(f"[预览下载响应] 状态码: {response.status_code}")
                    logger.info(f"[内容类型] {response.headers.get('Content-Type', 'Unknown')}")
                    logger.info(f"[内容长度] {response.headers.get('Content-Length', 'Unknown')} bytes")
                    
                    response.raise_for_status()
                    
                    # 创建输出目录
                    output_dir = os.path.join(os.path.dirname(__file__), "output", "videos")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 生成文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"seedream_video_{timestamp}.mp4"
                    local_path = os.path.join(output_dir, filename)
                    
                    # 保存视频
                    downloaded_size = 0
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                    
                    logger.info(f"[预览下载完成] 视频已保存到: {local_path}")
                    logger.info(f"[文件大小] {downloaded_size} bytes")
                    print(f"视频已保存到: {local_path}")
                    
                except requests.exceptions.HTTPError as e:
                    logger.error(f"[预览HTTP错误] {str(e)}")
                    logger.error(f"[响应状态] {e.response.status_code if e.response else 'No response'}")
                    logger.error(f"[响应内容] {e.response.text[:500] if e.response else 'No response'}")
                    print(f"下载视频失败: {str(e)}")
                    local_path = video_url
                    
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"[预览连接错误] {str(e)}")
                    logger.error(f"[URL] {video_url}")
                    print(f"下载视频失败: 连接错误 - {str(e)}")
                    local_path = video_url
                    
                except requests.exceptions.Timeout as e:
                    logger.error(f"[预览超时错误] {str(e)}")
                    logger.error(f"[URL] {video_url}")
                    print(f"下载视频失败: 请求超时 - {str(e)}")
                    local_path = video_url
                    
                except Exception as e:
                    logger.error(f"[预览未知错误] {type(e).__name__}: {str(e)}")
                    logger.error(f"[URL] {video_url}")
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