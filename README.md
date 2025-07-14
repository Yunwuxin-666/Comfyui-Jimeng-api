# 即梦AI ComfyUI节点

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

基于火山引擎API的ComfyUI自定义节点，提供图片生成、视频生成和智能对话功能。

## 📋 目录

- [功能特性](#功能特性)
- [安装方法](#安装方法)
- [使用说明](#使用说明)
- [API配置](#api配置)
- [节点详解](#节点详解)
- [工作流示例](#工作流示例)
- [故障排除](#故障排除)
- [更新日志](#更新日志)
- [支持与反馈](#支持与反馈)

## 🚀 功能特性

### 🖼️ 图片生成 (v1.0.4)
- **即梦图片生成**：调用火山引擎API生成高质量图片
  - 支持多种图片尺寸和比例
  - 可控制是否添加水印
  - 支持自定义随机种子
  - 高质量输出，适合专业创作

### 🎬 视频生成 (v2.6.0)
- **即梦视频生成**：调用火山引擎API生成视频
  - 支持文生视频、图生视频、首尾帧控制三种模式
  - 可选5秒或10秒时长
  - 支持多种画面比例
  - 可控制是否添加水印
  - 输出视频帧序列，可直接连接ComfyUI原生视频保存节点

#### 支持的模型
- **doubao-seedance-1-0-pro-250528**：全功能模型，支持所有生成模式
- **doubao-seedance-1-0-lite-t2v-250428**：轻量级文生视频模型（仅支持文本生成视频）
- **doubao-seedance-1-0-lite-i2v-250428**：轻量级图生视频模型（需要输入图片）

#### 生成模式
1. **文生视频**：仅需输入文本提示词
2. **图生视频**：输入首帧图片 + 文本提示词
3. **首尾帧控制**：输入首帧 + 尾帧图片，生成平滑过渡视频

### 🤖 智能对话 (v3.0.0)
- **豆包大语言模型**：使用最新的doubao-seed-1.6模型进行多模态对话
  - 支持深度思考模式（thinking/non-thinking/auto）
  - 支持文本、图像、视频输入
  - 最大256k上下文长度
  - 支持对话历史管理和上下文记忆

#### 核心特性
1. **深度思考**：适合复杂推理、数学计算、代码分析等场景
2. **多模态理解**：可以理解图片内容、分析视频场景
3. **对话管理**：支持保存/加载对话历史，维持长对话上下文

#### 使用场景
- 图像描述和分析
- 视频内容理解
- 专业知识问答
- 创意写作辅助
- 代码分析和调试

## 📦 安装方法

### 方法一：ComfyUI Manager安装（暂未上线，后续会支持上）
1. 在ComfyUI中打开Manager
2. 搜索"即梦AI"或"Jimeng"
3. 点击安装并重启ComfyUI

### 方法二：Git克隆安装
```bash
# 进入ComfyUI的custom_nodes目录
cd /path/to/ComfyUI/custom_nodes

# 克隆本仓库
git clone https://github.com/Yunwuxin-666/Comfyui-Jimeng-api.git

# 进入插件目录
cd Comfyui-Jimeng-api

# 安装依赖
pip install -r requirements.txt
```

### 方法三：手动安装
1. 下载本仓库的ZIP文件
2. 解压到ComfyUI的`custom_nodes`目录下
3. 确保文件夹名为`Comfyui-Jimeng-api`
4. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
5. 重启ComfyUI

### 依赖要求
- Python 3.8+
- ComfyUI
- 其他依赖详见 `requirements.txt`

## 🔧 API配置

### 获取API凭证

使用本插件需要火山引擎的API凭证：

**获取步骤：**
1. 访问[火山引擎控制台](https://console.volcengine.com/)
2. 注册并登录账号
3. 前往[密钥管理页面](https://console.volcengine.com/iam/keymanage/)
4. 创建或获取 **Access Key ID**（即API Key）

**重要提醒：**
- 请妥善保管API Key，不要泄露给他人
- 建议为不同项目使用不同的API Key
- 定期更换API Key以确保安全

## 📚 节点详解

### 1. 即梦图片生成节点

#### 节点参数
| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| `prompt` | STRING | 图片描述提示词 | 必填 |
| `api_key` | STRING | 火山引擎API密钥 | 必填 |
| `size` | COMBO | 图片比例和尺寸 | 1:1 (1024x1024) |
| `seed` | INT | 随机种子 | -1 |
| `watermark` | BOOLEAN | 是否添加水印 | false |

#### 支持的尺寸
- 1:1 (1024x1024) - 正方形
- 16:9 (1024x576) - 横屏
- 9:16 (576x1024) - 竖屏
- 4:3 (1024x768) - 标准
- 3:4 (768x1024) - 竖版标准

#### 使用示例
```
提示词: 鱼眼镜头，一只猫咪的头部，画面呈现出猫咪的五官因为拍摄方式扭曲的效果。
尺寸: 1:1 (1024x1024)
种子: -1 (随机)
水印: false
```

### 2. 即梦视频生成节点

#### 节点参数
| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| `prompt` | STRING | 视频描述文本 | 必填 |
| `api_key` | STRING | 火山引擎API密钥 | 必填 |
| `model_selection` | COMBO | 模型选择 | doubao-seedance-1-0-pro-250528 |
| `duration` | COMBO | 视频时长 | 5秒 |
| `ratio` | COMBO | 视频比例 | 16:9 |
| `watermark` | BOOLEAN | 是否添加水印 | false |
| `image` | IMAGE | 首帧图片（可选） | - |
| `end_image` | IMAGE | 尾帧图片（可选） | - |
| `seed` | INT | 随机种子 | -1 |

#### 支持的视频比例
- 21:9 - 超宽屏
- 16:9 - 横屏格式（推荐）
- 4:3 - 标准
- 1:1 - 正方形
- 3:4 - 竖版标准
- 9:16 - 竖屏格式（适合手机）
- 9:21 - 超高竖版
- keep_ratio - 保持原图比例（仅图生视频）
- adaptive - 自适应

#### 使用示例
```
提示词: 镜头缓慢推进，展现一片宁静的森林，阳光透过树叶洒下斑驳的光影
模型: doubao-seedance-1-0-pro-250528
时长: 5秒
比例: 16:9
水印: false
```

### 3. 豆包大语言模型节点

#### 节点参数
| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| `prompt` | STRING | 对话文本 | 必填 |
| `api_key` | STRING | 火山引擎API密钥 | 必填 |
| `thinking_mode` | COMBO | 思考模式 | auto |
| `image` | IMAGE | 输入图片（可选） | - |
| `video` | VIDEO | 输入视频（可选） | - |

#### 思考模式
- **thinking**：启用深度思考，适合复杂推理
- **non-thinking**：快速响应模式
- **auto**：自动选择合适的模式

## 🎯 工作流示例

### 图片生成工作流
1. 添加"即梦图片生成"节点
2. 输入提示词和API Key
3. 选择图片尺寸和参数
4. 连接到"Preview Image"节点查看结果
5. 可选连接到"Save Image"节点保存图片

### 视频生成工作流
1. 添加"即梦视频生成"节点
2. 输入提示词和API Key
3. 选择视频参数（时长、比例等）
4. 连接到"VHS_VideoCombine"节点保存为MP4
5. 或连接到其他视频处理节点

### 图生视频工作流
1. 使用"Load Image"节点加载图片
2. 连接到"即梦视频生成"节点的image输入
3. 设置提示词和参数
4. 输出视频帧序列

### 首尾帧控制工作流
1. 准备首帧和尾帧图片
2. 分别连接到"即梦视频生成"节点的image和end_image输入
3. 设置提示词描述过渡效果
4. 生成平滑过渡视频

## 📊 API限制说明

### 图片生成
- **限流**：500 IPM (每分钟最多生成500张图片)
- **定价**：0.259元/张
- **输出**：每次调用生成1张图片

### 视频生成
- **处理方式**：异步任务，需要等待处理完成
- **生成时间**：通常1-3分钟，复杂场景可能更长
- **建议**：高峰期适当延长等待时间

### 智能对话
- **上下文**：最大256k token
- **多模态**：支持文本、图像、视频输入
- **响应时间**：通常1-5秒

## 🔧 故障排除

### 通用问题
| 问题 | 原因 | 解决方法 |
|------|------|----------|
| API Key错误 | 密钥无效或过期 | 确保API Key正确且有效 |
| 网络错误 | 无法访问API | 检查网络连接，确保能访问火山引擎API |
| 依赖缺失 | 缺少必要的Python包 | 运行 `pip install -r requirements.txt` |

### 图片生成问题
| 问题 | 症状 | 解决方法 |
|------|------|----------|
| 限流错误 | 生成失败，提示限流 | 降低生成频率，增加延迟 |
| 尺寸参数错误 | `The parameter 'size' specified in the request is not valid` | 节点会自动处理，使用默认1024x1024 |

### 视频生成问题
| 问题 | 原因 | 解决方法 |
|------|------|----------|
| API认证失败 | API Key无效或无权限 | 确保API Key正确且有视频生成权限 |
| 任务创建失败 | 参数错误 | 检查提示词格式和模型名称 |
| 生成超时 | 处理时间过长 | 耐心等待，复杂场景需要更长时间 |
| 下载失败 | 本地问题 | 检查磁盘空间和写入权限 |

### 提示词优化建议
- 使用详细、具体的描述
- 参考专业的视频拍摄术语
- 描述运动、光线、镜头等细节
- 避免过于复杂或矛盾的描述

## 📝 更新日志

### v3.3.0 (最新)
- 🗑️ 移除SeedEdit功能
- 🎯 精简插件功能，专注于核心能力
- 🔧 优化代码结构

### v3.0.0
- 🆕 新增豆包大语言模型节点
- 🧠 支持深度思考模式
- 🖼️ 支持多模态输入（文本、图像、视频）
- 💬 支持对话历史管理

### v2.6.0
- 🎬 完善视频生成功能
- 🔄 优化异步处理流程
- 📊 改进错误处理和用户反馈

### v2.5.0
- 🎞️ 新增首尾帧控制功能
- 🔗 支持首尾帧输入，生成平滑过渡视频
- 📝 添加首尾帧控制示例工作流

### v2.4.0
- 🖼️➡️🎬 新增图生视频功能
- 📷 支持图片输入，实现图生视频
- 🔄 图片自动转换为base64格式
- 📐 新增更多视频比例选项

<details>
<summary>查看更多历史版本</summary>

### v2.3.1
- 🐛 修复参数传递问题
- 🔧 修复 "unexpected keyword argument" 错误

### v2.3.0
- 🎛️ 新增模型选择和水印控制
- 🏷️ 添加model_selection参数
- 💧 添加watermark参数

### v2.2.0
- 🎬 视频生成输出优化
- 🖼️ 输出帧序列（IMAGE格式）
- 🔗 支持直接连接ComfyUI视频保存节点

### v2.1.0
- 🎯 简化视频生成流程
- 🔧 无需创建推理接入点
- 🌐 使用专用的视频生成API端点

### v2.0.0
- 🆕 新增视频生成功能
- 🎬 添加即梦视频生成节点
- 📺 添加视频预览和保存功能

### v1.0.4
- 🎯 精简节点功能
- 🗑️ 移除批量生成节点
- 🔧 优化代码结构

### v1.0.3
- 🎨 进一步界面简化
- 🔢 移除图片数量参数
- 👥 优化用户体验

### v1.0.2
- 🎨 界面优化
- 🔒 隐藏高级参数
- 🎯 简化用户界面

### v1.0.1
- 💧 水印控制更新
- 🔧 添加watermark参数控制
- 🚫 默认关闭水印

### v1.0.0
- 🎉 初始版本
- 🖼️ 添加即梦图片生成节点
- 🔧 优化节点注册机制

</details>

## 📁 示例文件

### 图片生成
- `example_workflow.json` - 基础图片生成工作流
- 包含节点连接和参数设置示例

### 视频生成
- `video_example_workflow.json` - 基础视频生成工作流
- `text2video_example_workflow.json` - 文生视频示例
- `image2video_workflow.json` - 图生视频示例
- `keyframe_video_workflow.json` - 首尾帧控制示例
- `video_save_workflow.json` - 视频生成并保存为MP4的完整工作流

### 智能对话
- `chat_example_workflow.json` - 基础对话工作流
- `multimodal_chat_workflow.json` - 多模态对话示例

## 🤝 支持与反馈

### 获取帮助
- 📖 查看[Wiki文档](../../wiki)
- 🐛 提交[Issue](../../issues)
- 💬 参与[Discussions](../../discussions)

### 贡献代码
1. Fork本仓库
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

### 联系方式
- 📧 邮箱：[your-email@example.com](mailto:your-email@example.com)
- 🐙 GitHub：[@your-username](https://github.com/your-username)

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

感谢以下项目和开发者：
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的AI工作流平台
- [火山引擎](https://www.volcengine.com/) - 提供优质的AI服务
- 所有贡献者和用户的支持

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！