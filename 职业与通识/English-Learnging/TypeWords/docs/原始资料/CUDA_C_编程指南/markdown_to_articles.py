#!/usr/bin/env python3
"""
将 Markdown 文件转换为文章 JSON 格式
支持分批处理和进度记录
"""

import json
import re
import os
import random
import string
from pathlib import Path
from typing import List, Dict, Tuple

def generate_id(length=6):
    """生成随机 ID"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def clean_text(text: str) -> str:
    """清理文本"""
    # 移除多余的空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    # 移除页面标记
    text = re.sub(r'## Page \d+\n+', '', text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """将文本分割成句子"""
    # 按句号、问号、感叹号分割
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

def split_into_paragraphs(text: str) -> List[str]:
    """将文本分割成段落"""
    # 按双换行符分割
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def extract_title_from_section(text: str) -> str:
    """从文本中提取标题"""
    # 查找第一个标题行（以 # 开头）
    lines = text.split('\n')
    for line in lines[:10]:  # 只检查前10行
        line = line.strip()
        if line.startswith('#'):
            # 移除 # 标记
            title = re.sub(r'^#+\s*', '', line)
            if title and len(title) < 100:
                return title
    # 如果没有找到标题，使用第一句
    first_sentence = text.split('.')[0].strip()
    if len(first_sentence) < 100:
        return first_sentence[:80] + '...' if len(first_sentence) > 80 else first_sentence
    return "CUDA Programming Guide Section"

def translate_text(text: str) -> str:
    """
    翻译文本（简化版本，实际应该调用翻译 API）
    这里先返回占位符，后续可以集成翻译服务
    """
    # TODO: 集成翻译 API
    return f"[待翻译] {text}"

def format_text_for_article(paragraphs: List[str]) -> str:
    """格式化文本为文章格式（段落用 \n\n，句子用 \n）"""
    formatted_paragraphs = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        formatted_para = ' \n'.join(sentences)
        formatted_paragraphs.append(formatted_para)
    return '\n\n'.join(formatted_paragraphs)

def parse_markdown_sections(md_file: Path) -> List[Dict]:
    """解析 Markdown 文件，按页面提取内容"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    current_page = None
    current_text = []
    page_num = 0
    
    lines = content.split('\n')
    
    for line in lines:
        # 检测页面标记
        if line.startswith('## Page'):
            # 保存之前的页面
            if current_page is not None and current_text:
                text = '\n'.join(current_text).strip()
                if text and len(text) > 100:  # 只保留有足够内容的页面
                    # 尝试提取标题
                    title = extract_title_from_text(text)
                    sections.append({
                        'title': title,
                        'text': text,
                        'page': current_page
                    })
            
            # 提取页码
            match = re.search(r'Page (\d+)', line)
            if match:
                current_page = int(match.group(1))
                current_text = []
        elif current_page is not None:
            # 跳过分隔线和空行
            if line.strip() and not line.strip().startswith('---'):
                current_text.append(line)
    
    # 保存最后一个页面
    if current_page is not None and current_text:
        text = '\n'.join(current_text).strip()
        if text and len(text) > 100:
            title = extract_title_from_text(text)
            sections.append({
                'title': title,
                'text': text,
                'page': current_page
            })
    
    return sections

def extract_title_from_text(text: str) -> str:
    """从文本中提取标题"""
    lines = text.split('\n')
    for line in lines[:15]:  # 检查前15行
        line = line.strip()
        # 查找章节标题模式
        if re.match(r'^Chapter \d+\.', line, re.IGNORECASE):
            return line
        if re.match(r'^\d+\.\d+\.', line):  # 子章节如 3.1, 3.2
            return line
        if line and len(line) < 100 and line[0].isupper() and not line.startswith('CUDA'):
            # 可能是标题
            if len(line.split()) < 10:  # 标题通常单词较少
                return line
    # 如果没有找到，使用第一句
    first_line = lines[0].strip() if lines else ""
    if first_line and len(first_line) < 100:
        return first_line[:80] + '...' if len(first_line) > 80 else first_line
    return f"CUDA Programming Guide - Page Content"

def create_article_from_section(section: Dict, use_translation_api: bool = False) -> Dict:
    """从章节创建文章对象"""
    text = clean_text(section['text'])
    
    if not text or len(text) < 50:
        return None
    
    # 提取标题
    title = section.get('title', extract_title_from_section(text))
    if not title or title == '':
        title = "CUDA Programming Guide Section"
    
    # 分割段落
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return None
    
    # 格式化文本
    formatted_text = format_text_for_article(paragraphs)
    
    # 翻译（这里先用占位符）
    if use_translation_api:
        title_translate = translate_text(title)
        text_translate = translate_text(formatted_text)
    else:
        # 暂时使用占位符，后续可以手动翻译或使用 API
        title_translate = f"[待翻译] {title}"
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

def load_progress(progress_file: Path) -> Dict:
    """加载进度"""
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "status": "initialized",
        "articles_created": 0,
        "last_section_index": -1,
        "articles": []
    }

def save_progress(progress_file: Path, progress: Dict):
    """保存进度"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def main():
    base_dir = Path(__file__).parent
    md_file = base_dir / "CUDA_C_Programming_Guide.md"
    progress_file = base_dir / "article_generation_progress.json"
    output_json = base_dir / "CUDA_C_Programming_Guide_articles.json"
    
    # 加载进度
    progress = load_progress(progress_file)
    start_index = progress.get('last_section_index', -1) + 1
    
    print(f"开始处理 Markdown 文件: {md_file}")
    print(f"从第 {start_index + 1} 个章节开始...")
    
    # 解析 Markdown
    sections = parse_markdown_sections(md_file)
    print(f"共找到 {len(sections)} 个章节")
    
    # 加载已有文章
    articles = progress.get('articles', [])
    
    # 处理章节（每次处理 20 个）
    batch_size = 20
    end_index = min(start_index + batch_size, len(sections))
    
    print(f"\n处理章节 {start_index + 1} 到 {end_index}...")
    
    for i in range(start_index, end_index):
        section = sections[i]
        print(f"  处理章节 {i + 1}/{len(sections)}: {section.get('title', 'Unknown')[:50]}...")
        
        article = create_article_from_section(section)
        if article:
            articles.append(article)
            print(f"    ✓ 创建文章: {article['title'][:50]}...")
        else:
            print(f"    ✗ 跳过（内容太短）")
    
    # 保存进度
    progress['last_section_index'] = end_index - 1
    progress['articles_created'] = len(articles)
    progress['articles'] = articles
    progress['status'] = 'processing' if end_index < len(sections) else 'completed'
    
    save_progress(progress_file, progress)
    
    # 保存文章 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 处理完成！")
    print(f"  已创建文章: {len(articles)} 篇")
    print(f"  已处理章节: {end_index}/{len(sections)}")
    print(f"  输出文件: {output_json}")
    
    if end_index < len(sections):
        print(f"\n还有 {len(sections) - end_index} 个章节未处理，请再次运行脚本继续处理")
    else:
        print("\n所有章节已处理完成！")

if __name__ == "__main__":
    main()
