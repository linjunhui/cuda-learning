#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF提取和拆分脚本
将PDF文件提取为markdown格式，并按章节拆分
"""

import fitz  # PyMuPDF
import re
import os
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """从PDF中提取文本"""
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text.append(text)
    
    doc.close()
    return "\n".join(full_text)


def clean_text(text):
    """清理文本，移除多余空白"""
    # 移除多个连续的空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


def detect_sections(text):
    """检测章节标题，用于拆分"""
    lines = text.split('\n')
    sections = []
    current_section = []
    current_title = None
    
    # 常见的标题模式
    title_patterns = [
        r'^第[一二三四五六七八九十\d]+[章节部分]',  # 第一章、第一节等
        r'^\d+[\.、]\s*[^\n]{1,50}$',  # 1. 标题、1、标题
        r'^[一二三四五六七八九十]+[\.、]\s*[^\n]{1,50}$',  # 一、标题
        r'^[（(]\d+[）)]\s*[^\n]{1,50}$',  # (1) 标题
    ]
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if current_section:
                current_section.append('')
            continue
        
        # 检查是否是标题
        is_title = False
        for pattern in title_patterns:
            if re.match(pattern, line):
                is_title = True
                break
        
        # 如果行较短且后面跟着空行，可能是标题
        if not is_title and len(line) < 50 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line or (len(next_line) > 0 and not next_line[0].isdigit()):
                # 检查是否像标题（短行，可能包含关键词）
                title_keywords = ['指南', '方法', '技巧', '原则', '步骤', '建议', '注意', 
                                '总结', '引言', '前言', '概述', '介绍', '什么是', '如何']
                if any(keyword in line for keyword in title_keywords):
                    is_title = True
        
        if is_title and current_section:
            # 保存当前章节
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append({
                    'title': current_title or '未命名章节',
                    'content': section_text
                })
            current_section = [line]
            current_title = line
        else:
            current_section.append(line)
            if not current_title and line:
                # 第一个非空行作为标题
                if len(line) < 100:
                    current_title = line
    
    # 添加最后一个章节
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            sections.append({
                'title': current_title or '未命名章节',
                'content': section_text
            })
    
    return sections


def split_by_size(text, max_chars=5000):
    """按大小拆分文本（如果无法识别章节）"""
    sections = []
    lines = text.split('\n')
    current_section = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > max_chars and current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                # 尝试从当前section中找到标题
                title = '部分内容'
                for title_line in current_section[:5]:
                    if title_line.strip() and len(title_line.strip()) < 50:
                        title = title_line.strip()
                        break
                
                sections.append({
                    'title': title,
                    'content': section_text
                })
            current_section = []
            current_size = 0
        
        current_section.append(line)
        current_size += line_size
    
    # 添加最后一个section
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            title = '部分内容'
            for title_line in current_section[:5]:
                if title_line.strip() and len(title_line.strip()) < 50:
                    title = title_line.strip()
                    break
            sections.append({
                'title': title,
                'content': section_text
            })
    
    return sections


def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    # 移除或替换非法字符
    illegal_chars = r'[<>:"/\\|?*]'
    filename = re.sub(illegal_chars, '_', filename)
    # 限制长度
    if len(filename) > 100:
        filename = filename[:100]
    return filename


def save_sections(sections, output_dir):
    """保存章节到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for i, section in enumerate(sections, 1):
        # 生成文件名
        title = section['title']
        safe_title = sanitize_filename(title)
        filename = f"{i:02d}_{safe_title}.md"
        filepath = os.path.join(output_dir, filename)
        
        # 如果文件名太长，使用序号
        if len(filename) > 200:
            filename = f"{i:02d}_section_{i}.md"
            filepath = os.path.join(output_dir, filename)
        
        # 写入文件
        content = f"# {title}\n\n{section['content']}\n"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        saved_files.append(filepath)
        print(f"已保存: {filename}")
    
    return saved_files


def main():
    pdf_path = "2.17w字-社会化指南-悦悦.pdf"
    output_dir = "社会化指南-拆分"
    
    print(f"正在提取PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    text = clean_text(text)
    
    print(f"提取完成，总字符数: {len(text)}")
    print("正在检测章节...")
    
    # 尝试按章节拆分
    sections = detect_sections(text)
    
    # 如果章节太少，尝试按大小拆分
    if len(sections) < 3:
        print("章节检测结果较少，改为按大小拆分...")
        sections = split_by_size(text, max_chars=5000)
    
    print(f"检测到 {len(sections)} 个章节")
    
    # 保存章节
    print(f"正在保存到目录: {output_dir}")
    saved_files = save_sections(sections, output_dir)
    
    # 同时保存完整版本
    full_file = os.path.join(output_dir, "00_完整版.md")
    with open(full_file, 'w', encoding='utf-8') as f:
        f.write(f"# 社会化指南 - 完整版\n\n{text}\n")
    print(f"已保存完整版: {full_file}")
    
    print(f"\n完成！共生成 {len(saved_files) + 1} 个文件")


if __name__ == "__main__":
    main()

