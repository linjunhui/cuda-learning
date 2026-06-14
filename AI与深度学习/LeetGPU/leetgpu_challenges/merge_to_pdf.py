#!/usr/bin/env python3
"""
将当前目录下的所有 Markdown 文件合并成一个 PDF
- 带目录
- 每个 md 文件占一页
"""

import os
import glob
import subprocess
import sys
from pathlib import Path

def get_md_files(directory):
    """获取目录下所有的 md 文件，排除合并转pdf.md"""
    md_files = []
    for file in glob.glob(os.path.join(directory, "*.md")):
        filename = os.path.basename(file)
        if filename != "合并转pdf.md" and filename != "merge_to_pdf.py":
            md_files.append(file)
    
    # 按文件名排序（数字开头的会按数字大小排序）
    def sort_key(f):
        basename = os.path.basename(f)
        # 提取文件名开头的数字
        try:
            num = int(basename.split('_')[0])
            return (0, num)
        except ValueError:
            return (1, basename)
    
    md_files.sort(key=sort_key)
    return md_files

def merge_md_files(md_files, output_file):
    """合并所有 md 文件，每个文件之间添加分页符"""
    merged_content = []
    
    for i, md_file in enumerate(md_files):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 添加文件内容
        merged_content.append(content)
        
        # 在每个文件后添加分页符（除了最后一个文件）
        if i < len(md_files) - 1:
            merged_content.append('\n\n\\newpage\n\n')
    
    # 写入临时合并文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(merged_content))
    
    print(f"已合并 {len(md_files)} 个文件到 {output_file}")

def convert_to_pdf(input_file, output_pdf):
    """使用 pandoc 将合并的 md 文件转换为 PDF"""
    # 检查 pandoc 是否安装
    try:
        subprocess.run(['pandoc', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未找到 pandoc。请先安装 pandoc:")
        print("  Ubuntu/Debian: sudo apt-get install pandoc texlive-xetex texlive-lang-chinese")
        print("  或者: sudo apt-get install pandoc texlive-full")
        sys.exit(1)
    
    # 检查是否安装了 LaTeX（用于生成 PDF）
    try:
        subprocess.run(['xelatex', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: 未找到 xelatex。尝试使用 pdflatex...")
        latex_engine = 'pdflatex'
    else:
        latex_engine = 'xelatex'
    
    # 构建 pandoc 命令
    cmd = [
        'pandoc',
        input_file,
        '-o', output_pdf,
        '--pdf-engine=' + latex_engine,
        '--toc',  # 生成目录
        '--toc-depth=2',  # 目录深度
        '-V', 'geometry:margin=2cm',  # 设置页边距
        '-V', 'CJKmainfont=SimSun',  # 中文字体（如果系统有的话）
        '--highlight-style=tango',  # 代码高亮样式
    ]
    
    print(f"正在转换为 PDF: {output_pdf}")
    print(f"使用引擎: {latex_engine}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ 成功生成 PDF: {output_pdf}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: PDF 转换失败")
        print(f"错误信息: {e.stderr}")
        return False

def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("正在查找 Markdown 文件...")
    md_files = get_md_files(script_dir)
    
    if not md_files:
        print("未找到任何 Markdown 文件")
        return
    
    print(f"找到 {len(md_files)} 个 Markdown 文件:")
    for f in md_files[:5]:  # 只显示前5个
        print(f"  - {os.path.basename(f)}")
    if len(md_files) > 5:
        print(f"  ... 还有 {len(md_files) - 5} 个文件")
    
    # 合并文件
    merged_md = 'merged_output.md'
    merge_md_files(md_files, merged_md)
    
    # 转换为 PDF
    output_pdf = 'merged_output.pdf'
    if convert_to_pdf(merged_md, output_pdf):
        # 询问是否删除临时文件
        try:
            os.remove(merged_md)
            print(f"已删除临时文件: {merged_md}")
        except:
            pass
        print(f"\n完成! PDF 文件已生成: {output_pdf}")
    else:
        print(f"\n转换失败，但合并的 Markdown 文件已保存: {merged_md}")

if __name__ == '__main__':
    main()
























