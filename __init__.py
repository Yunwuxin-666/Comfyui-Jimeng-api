"""
即梦API ComfyUI节点
版本: v3.3.0
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 从各个模块导入节点类
from .seedream_node import SeedreamImageGeneratorNode
from .seedream_video_node import SeedreamVideoGeneratorNode
from .seedchat import DoubaoLLMNode
from .seededit import SeedEditImageEditorNode
from .seedream4_0 import Seedream4StreamingNode

# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "SeedreamImageGeneratorNode": SeedreamImageGeneratorNode,
    "SeedreamVideoGeneratorNode": SeedreamVideoGeneratorNode,
    "DoubaoLLMNode": DoubaoLLMNode,
    "SeedEditImageEditorNode": SeedEditImageEditorNode,
    "Seedream4StreamingNode": Seedream4StreamingNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedreamImageGeneratorNode": "即梦图像生成",
    "SeedreamVideoGeneratorNode": "即梦视频生成",
    "DoubaoLLMNode": "豆包大语言模型",
    "SeedEditImageEditorNode": "即梦图像编辑",
    "Seedream4StreamingNode": "即梦4.0生成",
}

# 导出列表
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 