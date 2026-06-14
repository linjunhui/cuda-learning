#!/usr/bin/env python3
"""
PDF 转 Markdown 转换脚本
支持分批处理和进度记录
"""

import sys
import json
import os
from pathlib import Path
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, start_page=0, end_page=None):
    """从 PDF 提取文本"""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    if end_page is None:
        end_page = total_pages
    
    end_page = min(end_page, total_pages)
    
    text_content = []
    for i in range(start_page, end_page):
        try:
            page = reader.pages[i]
            text = page.extract_text()
            text_content.append({
                'page': i + 1,
                'text': text
            })
        except Exception as e:
            print(f"Error extracting page {i+1}: {e}", file=sys.stderr)
    
    return text_content, total_pages

def save_progress(progress_file, current_page, total_pages, output_file):
    """保存进度"""
    progress = {
        'current_page': current_page,
        'total_pages': total_pages,
        'output_file': output_file,
        'status': 'processing'
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def main():
    pdf_path = Path(__file__).parent / "CUDA_C_Programming_Guide.pdf"
    output_md = Path(__file__).parent / "CUDA_C_Programming_Guide.md"
    progress_file = Path(__file__).parent / "conversion_progress.json"
    
    # 检查是否已有进度
    start_page = 0
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                start_page = progress.get('current_page', 0)
                print(f"从第 {start_page + 1} 页继续转换...")
        except:
            pass
    
    # 提取文本（每次处理 50 页）
    pages_per_batch = 50
    all_text = []
    
    if start_page == 0:
        # 第一次运行，创建新文件
        mode = 'w'
    else:
        # 继续处理，追加内容
        mode = 'a'
        # 读取已有内容
        if output_md.exists():
            with open(output_md, 'r', encoding='utf-8') as f:
                existing = f.read()
                if existing.strip():
                    all_text.append(existing)
    
    print(f"开始转换 PDF: {pdf_path}")
    print(f"从第 {start_page + 1} 页开始...")
    
    text_content, total_pages = extract_text_from_pdf(pdf_path, start_page)
    
    # 转换为 Markdown 格式
    md_content = []
    if mode == 'w':
        md_content.append("# CUDA C Programming Guide\n\n")
        md_content.append(f"*Total Pages: {total_pages}*\n\n")
        md_content.append("---\n\n")
    
    for page_data in text_content:
        page_num = page_data['page']
        text = page_data['text']
        
        # 清理文本
        text = text.strip()
        if not text:
            continue
        
        # 添加页面标记
        md_content.append(f"\n## Page {page_num}\n\n")
        md_content.append(text)
        md_content.append("\n\n---\n\n")
        
        # 更新进度
        save_progress(progress_file, page_num, total_pages, str(output_md))
    
    # 保存 Markdown
    with open(output_md, mode, encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    current_max_page = max([p['page'] for p in text_content]) if text_content else start_page
    
    print(f"\n转换完成！")
    print(f"已处理: {current_max_page}/{total_pages} 页")
    print(f"输出文件: {output_md}")
    
    if current_max_page < total_pages:
        print(f"\n还有 {total_pages - current_max_page} 页未处理，请再次运行脚本继续转换")
    else:
        print("\n所有页面已转换完成！")
        # 标记完成
        progress = {
            'current_page': total_pages,
            'total_pages': total_pages,
            'output_file': str(output_md),
            'status': 'completed'
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
