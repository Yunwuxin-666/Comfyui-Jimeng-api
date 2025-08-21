# 火山引擎即梦AI生成 ComfyUI 节点包

![Version](https://img.shields.io/badge/version-1.2.0-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green)

## 🎯 核心功能

本节点包为 ComfyUI 提供了火山引擎即梦AI的四个核心功能：

### 1. 🎨 图像生成 (SeedreamImageGeneratorNode)
- **功能**: 根据文本描述生成高质量图像
- **模型**: doubao-seedream-3.0-t2i
- **支持**: 多种尺寸比例，自定义种子，水印控制

### 2. ✏️ 图像编辑 (SeedEditImageEditorNode)  
- **功能**: 智能编辑现有图像
- **支持**: 局部重绘、背景替换、风格转换、超分辨率等
- **模型**: doubao-seededit-3-0-i2i-250628

### 3. 🎬 视频生成 (SeedreamVideoGeneratorNode)
- **功能**: 从文本或图像生成动态视频
- **模式**: 文生视频、图生视频、首尾帧控制
- **模型**: doubao-seedance-1-0-pro-250528

### 4. 🤖 大语言模型 (DoubaoLLMNode)
- **功能**: 智能文本生成和图像理解
- **支持**: 多模态输入、思维链推理、流式输出
- **模型**: doubao-pro-4k

## 🚀 快速开始

### 安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/Comfyui-Jimeng-api.git
```

### 配置API Key
在插件目录下创建 `apikey.txt` 文件：
```
your-api-key-here
```

### 使用
1. 在ComfyUI中添加对应节点
2. 连接输入输出
3. 填写提示词和参数
4. 点击执行生成

## 📖 参数说明

### 通用参数
- **api_key**: 火山引擎API密钥
- **watermark**: 水印控制 (false/true)
- **seed**: 随机种子 (-1为随机)

### 图像生成
- **prompt**: 图像描述文本
- **size**: 图像尺寸 (多种比例可选)

### 图像编辑  
- **image**: 输入图像
- **prompt**: 编辑指令
- **strength**: 编辑强度 (0-1)
- **mask**: 编辑区域蒙版

### 视频生成
- **prompt**: 视频描述
- **duration**: 视频时长 (5/10秒)
- **ratio**: 视频比例
- **fps**: 帧率 (16/24)

## 🔄 工作流示例

### 完整AI创作流程
```
[文本输入] → [图像生成] → [图像编辑] → [视频生成]
     ↓           ↓           ↓           ↓
[LLM理解] → [生成图像] → [智能编辑] → [动态视频]
```

## 🔗 相关链接

- [火山引擎官网](https://www.volcengine.com/)
- [即梦AI文档](https://www.volcengine.com/docs/82379)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)

## 📝 更新日志

### v1.2.0 (2024-01-25)
- 新增大语言模型节点
- 优化网络连接稳定性
- 完善水印控制功能
- 简化参数配置

### v1.1.0 (2024-01-20)
- 新增图像编辑节点
- 优化网络请求稳定性

### v1.0.0 (2024-01-15)
- 初始版本发布
- 支持图像和视频生成

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License