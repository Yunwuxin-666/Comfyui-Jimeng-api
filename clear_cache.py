#!/usr/bin/env python3
"""
清理ComfyUI缓存脚本
用于在移除SeedEdit功能后清理缓存
"""

import os
import sys
import shutil
import importlib
import glob

def clear_pycache():
    """清理Python缓存文件"""
    print("正在清理Python缓存...")
    
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 清理__pycache__目录
    pycache_dir = os.path.join(current_dir, "__pycache__")
    if os.path.exists(pycache_dir):
        print(f"清理目录: {pycache_dir}")
        # 删除seededit开头的缓存文件
        for cache_file in glob.glob(os.path.join(pycache_dir, "seededit*.py[co]")):
            print(f"删除文件: {os.path.basename(cache_file)}")
            os.remove(cache_file)
        
        # 删除seededit开头的__pycache__目录
        for cache_dir in glob.glob(os.path.join(pycache_dir, "seededit*")):
            if os.path.isdir(cache_dir):
                print(f"删除目录: {os.path.basename(cache_dir)}")
                shutil.rmtree(cache_dir)
    else:
        print("__pycache__目录不存在，无需清理")
    
    # 清理sys.modules中的缓存
    modules_to_remove = []
    for module_name in sys.modules:
        if module_name.startswith("seededit") or "seededit" in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            print(f"从sys.modules中移除: {module_name}")
            del sys.modules[module_name]
    
    print("Python缓存清理完成")

if __name__ == "__main__":
    clear_pycache()
    print("\n所有SeedEdit相关缓存已清理完成。")
    print("请重启ComfyUI以应用更改。") 