#!/usr/bin/env python3
"""
按照原始章节结构提取文章
"""

import json
import re
import random
import string
from pathlib import Path

def generate_id(length=6):
    """生成随机 ID"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def clean_text(text):
    """清理文本"""
    # 移除页面标记
    text = re.sub(r'## Page \d+\n+', '', text)
    # 移除多余的空白行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除行尾的数字（页码）
    text = re.sub(r' \d+\n', '\n', text)
    return text.strip()

def split_into_sentences(text):
    """将文本分割成句子"""
    sentences = re.split(r'([.!?])\s+', text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result if result else [text]

def split_into_paragraphs(text):
    """将文本分割成段落"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def format_text_for_article(paragraphs):
    """格式化文本为文章格式（段落用 \n\n，句子用 \n）"""
    formatted_paragraphs = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        formatted_para = ' \n'.join(sentences)
        formatted_paragraphs.append(formatted_para)
    return '\n\n'.join(formatted_paragraphs)

def extract_chapters(md_file):
    """从Markdown文件中提取章节"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chapters = []
    lines = content.split('\n')
    
    current_chapter = None
    current_content = []
    chapter_start_line = None
    
    for i, line in enumerate(lines):
        # 检测章节标题（Chapter X.）
        if re.match(r'^Chapter \d+\.', line):
            # 保存之前的章节
            if current_chapter and current_content:
                chapter_text = '\n'.join(current_content)
                chapter_text = clean_text(chapter_text)
                if chapter_text and len(chapter_text) > 100:
                    chapters.append({
                        'title': current_chapter,
                        'text': chapter_text,
                        'start_line': chapter_start_line
                    })
            
            # 开始新章节
            current_chapter = line.strip()
            current_content = [line]
            chapter_start_line = i
        elif current_chapter:
            # 跳过页面标记和分隔线
            if not line.startswith('## Page') and not line.strip().startswith('---'):
                current_content.append(line)
    
    # 保存最后一个章节
    if current_chapter and current_content:
        chapter_text = '\n'.join(current_content)
        chapter_text = clean_text(chapter_text)
        if chapter_text and len(chapter_text) > 100:
            chapters.append({
                'title': current_chapter,
                'text': chapter_text,
                'start_line': chapter_start_line
            })
    
    return chapters

def create_article_from_chapter(chapter):
    """从章节创建文章对象"""
    text = chapter['text']
    title = chapter['title']
    
    # 提取标题（移除 "Chapter X." 前缀，保留后面的内容）
    title_clean = re.sub(r'^Chapter \d+\.\s*', '', title).strip()
    if not title_clean:
        title_clean = title
    
    # 分割段落
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return None
    
    # 格式化文本
    formatted_text = format_text_for_article(paragraphs)
    
    # 翻译标题（简化版，实际需要手动翻译）
    title_translate_map = {
        "Overview": "概述",
        "What Is the CUDA C Programming Guide?": "什么是 CUDA C 编程指南？",
        "Introduction": "简介",
        "Changelog": "更新日志",
        "Programming Model": "编程模型",
        "Programming Interface": "编程接口",
        "Hardware Implementation": "硬件实现",
        "Performance Guidelines": "性能指南",
        "CUDA-Enabled GPUs": "支持 CUDA 的 GPU",
        "C++ Language Extensions": "C++ 语言扩展",
        "Cooperative Groups": "协作组",
        "Cluster Launch Control": "集群启动控制",
        "CUDA Dynamic Parallelism": "CUDA 动态并行",
        "Virtual Memory": "虚拟内存",
        "Stream Ordered Memory": "流有序内存",
        "Graph Memory Nodes": "图内存节点",
        "Mathematical Functions": "数学函数",
        "C++ Language Support": "C++ 语言支持",
        "Texture Fetching": "纹理获取",
        "Compute Capabilities": "计算能力",
        "Driver API": "驱动 API",
        "CUDA Environment": "CUDA 环境",
        "Error Log Management": "错误日志管理",
        "Unified Memory": "统一内存",
        "Lazy Loading": "延迟加载",
        "Extended GPU Memory": "扩展 GPU 内存",
        "Notices": "声明"
    }
    
    title_translate = title_translate_map.get(title_clean, f"[待翻译] {title_clean}")
    
    # 翻译文本（占位符，需要手动翻译）
    text_translate = f"[待翻译] {formatted_text}"
    
    article = {
        "id": generate_id(6),
        "title": title,
        "titleTranslate": title_translate,
        "text": formatted_text,
        "textTranslate": text_translate,
        "newWords": [],
        "audioSrc": "",
        "audioFileId": "",
        "lrcPosition": [],
        "questions": [],
        "nameList": []
    }
    
    return article

def main():
    base_dir = Path(__file__).parent
    md_file = base_dir / "CUDA_C_Programming_Guide.md"
    # 找到项目根目录（包含 public 目录的目录）
    project_root = base_dir.parent.parent.parent
    output_json = project_root / "public/dicts/en/article/CUDA_C_Programming_Guide.json"
    article_list_file = project_root / "public/list/article.json"
    
    print(f"从 Markdown 文件提取章节: {md_file}")
    
    # 提取章节
    chapters = extract_chapters(md_file)
    print(f"找到 {len(chapters)} 个章节")
    
    # 创建文章
    articles = []
    for i, chapter in enumerate(chapters):
        print(f"处理章节 {i+1}/{len(chapters)}: {chapter['title'][:50]}...")
        article = create_article_from_chapter(chapter)
        if article:
            articles.append(article)
            print(f"  ✓ 创建文章: {article['title'][:50]}...")
    
    # 保存文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 完成！")
    print(f"  创建了 {len(articles)} 篇文章")
    print(f"  输出文件: {output_json}")
    
    # 更新 article.json
    if article_list_file.exists():
        with open(article_list_file, 'r', encoding='utf-8') as f:
            article_list = json.load(f)
        
        # 更新或添加书籍信息
        book_id = "article_cuda_programming_guide"
        book_found = False
        for book in article_list:
            if book.get('id') == book_id:
                book['length'] = len(articles)
                book_found = True
                break
        
        if not book_found:
            article_list.append({
                "id": book_id,
                "name": "CUDA C++ 编程指南",
                "description": "NVIDIA官方CUDA C++编程指南，涵盖CUDA架构、编程模型、语言扩展和性能优化等核心内容。",
                "category": "文章学习",
                "tags": ["CUDA", "编程", "GPU", "并行计算", "NVIDIA", "官方文档"],
                "url": "CUDA_C_Programming_Guide.json",
                "length": len(articles),
                "translateLanguage": "common",
                "language": "en",
                "cover": "",
                "update": True
            })
        
        with open(article_list_file, 'w', encoding='utf-8') as f:
            json.dump(article_list, f, ensure_ascii=False, indent=2)
        
        print(f"  已更新 article.json")

if __name__ == "__main__":
    main()
