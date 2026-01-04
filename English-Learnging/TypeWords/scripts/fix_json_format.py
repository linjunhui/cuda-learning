#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理JSON文件格式，确保符合规则：
1. 每个句号、问号、感叹号后面必须有 \n\n
2. 使用 \n\n 而不是单个 \n 或 \\n
"""

import json
import re
import os
from pathlib import Path


def fix_text_format(text: str) -> str:
    """
    修复文本格式，确保每个句子后面都有 \n\n
    核心规则：有句号、问号、感叹号就要加 \n\n
    """
    if not text:
        return text
    
    # 步骤1: 处理句子结尾标点后直接跟非空白字符的情况
    # 模式：标点 + 可选空格 + 非空白字符（说明直接开始下一句，需要添加 \n\n）
    # 但要注意不要匹配引号内的标点（这比较复杂，先简单处理）
    text = re.sub(r'([。！？.!?])\s+([^\s\n])', r'\1\n\n\2', text)
    
    # 步骤2: 处理句子结尾标点后跟单个 \n 的情况（应该改为 \n\n）
    # 模式：标点 + 可选空格 + 单个 \n + 非 \n 字符
    text = re.sub(r'([。！？.!?])\s*\n(?!\n)', r'\1\n\n', text)
    
    # 步骤3: 处理句子结尾标点后没有任何换行符，直接跟下一句的情况
    # 模式：标点 + 非空白非换行字符（这种情况应该很少，因为通常会有空格）
    # 但为了保险，还是处理一下
    text = re.sub(r'([。！？.!?])([^\s\n])', r'\1\n\n\2', text)
    
    # 步骤4: 清理连续的换行符（超过两个的）
    text = re.sub(r'\n{3,}', r'\n\n', text)
    
    # 步骤5: 移除文本末尾的多余空白
    text = text.rstrip()
    
    return text


def process_json_file(file_path: Path) -> bool:
    """
    处理单个JSON文件
    返回True如果文件被修改，False如果没有修改
    """
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        # 处理数组中的每个文章对象
        if isinstance(data, list):
            for article in data:
                if isinstance(article, dict):
                    # 处理 text 字段
                    if 'text' in article:
                        original = article['text']
                        fixed = fix_text_format(original)
                        if original != fixed:
                            article['text'] = fixed
                            modified = True
                    
                    # 处理 textTranslate 字段
                    if 'textTranslate' in article:
                        original = article['textTranslate']
                        fixed = fix_text_format(original)
                        if original != fixed:
                            article['textTranslate'] = fixed
                            modified = True
        
        # 如果文件被修改，保存回去
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 已修复: {file_path.name}")
            return True
        else:
            print(f"✓ 无需修改: {file_path.name}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误 {file_path.name}: {e}")
        return False
    except Exception as e:
        print(f"❌ 处理错误 {file_path.name}: {e}")
        return False


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    # 获取项目根目录（假设脚本在 scripts/ 目录下）
    project_root = script_dir.parent
    # 目标目录
    target_dir = project_root / 'dist' / 'dicts' / 'en' / 'article'
    
    if not target_dir.exists():
        print(f"❌ 目录不存在: {target_dir}")
        return
    
    # 获取所有JSON文件
    json_files = list(target_dir.glob('*.json'))
    
    if not json_files:
        print(f"❌ 在 {target_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件\n")
    
    modified_count = 0
    for json_file in sorted(json_files):
        if process_json_file(json_file):
            modified_count += 1
    
    print(f"\n完成！共处理 {len(json_files)} 个文件，修改了 {modified_count} 个文件")


if __name__ == '__main__':
    main()
