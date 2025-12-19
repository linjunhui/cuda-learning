"""
使用API批量获取所有LeetGPU题目的完整信息并保存为Markdown文件
文件名格式: {id}_{title-slug}.md

使用方法：
1. 如果token过期，请更新 AUTH_TOKEN 变量
2. 运行: python fetch_all_challenges.py
"""
import json
import os
import re
import html
from datetime import datetime
import urllib.request
import urllib.error

# API配置
API_URL = "https://api.leetgpu.com/api/v1/challenges/fetch-all"
AUTH_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsImtpZCI6IkZqdUp4N1hkZjFJaDkybzkiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3loZHR5c2FjZGtxb3F1dmtkd2RkLnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiJmYTYwOTMyNC03NGU2LTQ5NzgtOGE5Yi03ZGUxZTM1YzYzM2QiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzY2MDY1NjI4LCJpYXQiOjE3NjYwNjIwMjgsImVtYWlsIjoibGluMTU2MDI0NzEyNjlAZ21haWwuY29tIiwicGhvbmUiOiIiLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJnb29nbGUiLCJwcm92aWRlcnMiOlsiZ29vZ2xlIl19LCJ1c2VyX21ldGFkYXRhIjp7ImF2YXRhcl91cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJNDljbXF1SldWeHRHejljempXZWktSVVTbjhwc0pKNnNyTDVJNDR4V1hIYnVlbGc9czk2LWMiLCJlbWFpbCI6ImxpbjE1NjAyNDcxMjY5QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmdWxsX25hbWUiOiLmnpfkv4rmmZYiLCJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYW1lIjoi5p6X5L-K5pmWIiwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSTQ5Y21xdUpXVnh0R3o5Y3pqV2VpLUlVU244cHNKSjZzckw1STQ0eFdYSGJ1ZWxnPXM5Ni1jIiwicHJvdmlkZXJfaWQiOiIxMTE1Mjg2MDQxMjY3MzAwNjQ5NjAiLCJzdWIiOiIxMTE1Mjg2MDQxMjY3MzAwNjQ5NjAifSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvYXV0aCIsInRpbWVzdGFtcCI6MTc2NTUyNjM4MH1dLCJzZXNzaW9uX2lkIjoiN2U4YWE3N2QtNGE5MS00MzA2LWJkMTItZDQ5MDhkYjM1MDk1IiwiaXNfYW5vbnltb3VzIjpmYWxzZX0.pvCZwZVFyCvL-UVxUqeWZlPnFvdnxA3vFt8V-QIptM0"

OUTPUT_DIR = "leetgpu_challenges"

def html_to_markdown(html_text):
    """
    将HTML格式的题目描述转换为Markdown
    修复了所有HTML标签处理和格式问题
    """
    if not html_text:
        return ""
    
    # 1. 先解码HTML实体
    text = html.unescape(html_text)
    
    # 2. 处理行内HTML标签（在段落处理之前）
    # <strong>...</strong> -> **...**
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    
    # <em>...</em> -> *...*
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    
    # <sub>...</sub> -> 下标
    text = re.sub(r'<sub>(.*?)</sub>', r'_{\1}', text, flags=re.DOTALL)
    
    # <sup>...</sup> -> 上标
    text = re.sub(r'<sup>(.*?)</sup>', r'^{\1}', text, flags=re.DOTALL)
    
    # <br> / <br/> -> 换行（但需要小心，避免在代码块中破坏格式）
    # 先标记代码块区域
    code_blocks = []
    def mark_code(match):
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{idx}__"
    
    text = re.sub(r'<pre>.*?</pre>', mark_code, text, flags=re.DOTALL)
    
    # 处理换行标签（在代码块之外）
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # 恢复代码块
    for idx, code in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{idx}__", code)
    
    # 3. 处理带样式的标签（移除样式属性，只保留标签语义）
    text = re.sub(r'<(\w+)\s+[^>]*style=["\'][^"\']*["\'][^>]*>', r'<\1>', text)
    
    # 4. 处理代码标签（行内代码）
    text = re.sub(r'<code>', '`', text)
    text = re.sub(r'</code>', '`', text)
    
    # 5. 处理代码块（pre标签）- 保持代码块内的原始格式
    def process_pre(match):
        code_content = match.group(1)
        # 清理代码块内的多余缩进（保留相对缩进结构）
        lines = code_content.split('\n')
        # 移除首尾空行
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop(-1)
        # 找到最小缩进
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        # 移除统一的最小缩进
        if min_indent > 0 and min_indent != float('inf'):
            lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
        code_content = '\n'.join(lines)
        return f'\n```\n{code_content}\n```\n'
    
    text = re.sub(r'<pre>\s*(.*?)\s*</pre>', process_pre, text, flags=re.DOTALL)
    
    # 6. 处理段落标签
    text = re.sub(r'<p>\s*', '\n\n', text)
    text = re.sub(r'</p>', '', text)
    
    # 7. 处理标题
    text = re.sub(r'<h2>\s*', '\n\n## ', text)
    text = re.sub(r'</h2>', '', text)
    text = re.sub(r'<h3>\s*', '\n\n### ', text)
    text = re.sub(r'</h3>', '', text)
    text = re.sub(r'<h4>\s*', '\n\n#### ', text)
    text = re.sub(r'</h4>', '', text)
    
    # 8. 处理列表
    text = re.sub(r'<ul>\s*', '\n', text)
    text = re.sub(r'</ul>', '', text)
    text = re.sub(r'<ol>\s*', '\n', text)
    text = re.sub(r'</ol>', '', text)
    
    # 处理列表项，清理多余的缩进
    def process_li(match):
        content = match.group(1)
        # 移除列表项内容的前导空白
        content = content.strip()
        # 如果内容是多行，清理每行的缩进（但保留相对结构）
        if '\n' in content:
            lines = content.split('\n')
            # 找到最小缩进
            min_indent = float('inf')
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            # 移除统一的最小缩进
            if min_indent > 0 and min_indent != float('inf'):
                lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
            content = '\n'.join(lines)
        # 如果内容以"- "开头，说明是嵌套列表，需要处理
        if content.startswith('- '):
            # 嵌套列表项，保持一个缩进级别（2个空格）
            return f'  {content}\n'
        return f'- {content}\n'
    
    text = re.sub(r'<li>\s*(.*?)\s*</li>', process_li, text, flags=re.DOTALL)
    
    # 后处理：清理列表项后面的多余缩进行（这些是从HTML中来的缩进内容）
    lines = text.split('\n')
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        cleaned_lines.append(line)
        # 如果当前是列表项（以"- "开头且不是缩进的）
        if line.strip().startswith('- ') and not line.startswith('  '):
            # 检查接下来的行是否是缩进的内容（应该合并到当前列表项或作为嵌套列表）
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('-') and not lines[j].strip().startswith('*') and not lines[j].strip().startswith('#'):
                next_line = lines[j]
                # 如果下一行有缩进且不是代码块，说明是列表项内容的延续
                if next_line.startswith('    ') and not next_line.strip().startswith('```'):
                    # 移除多余的缩进，只保留2个空格作为列表项内容
                    cleaned_lines.pop()  # 移除刚添加的列表项
                    # 将缩进内容合并到上一行或作为独立行（移除缩进）
                    content_line = next_line[4:]  # 移除4个空格
                    if content_line.strip().startswith('- '):
                        # 如果内容是嵌套列表，保留2个空格缩进
                        cleaned_lines.append(line)  # 重新添加原列表项
                        cleaned_lines.append('  ' + content_line)
                    else:
                        # 否则作为普通内容，移除缩进
                        cleaned_lines.append(line)  # 重新添加原列表项
                        cleaned_lines.append(content_line)
                    j += 1
                    break
                else:
                    break
            i = j
        else:
            i += 1
    
    text = '\n'.join(cleaned_lines)
    
    # 9. 处理链接
    def replace_link(match):
        href = match.group(1)
        text_content = match.group(2)
        return f'[{text_content}]({href})'
    text = re.sub(r'<a\s+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', replace_link, text, flags=re.DOTALL)
    
    # 10. 清理格式问题
    # 移除连续的4个或更多美元符号
    text = re.sub(r'\$\$\s*\$\$', '$$', text)
    
    # 移除行首的额外空格（保留代码块和列表的缩进）
    lines = text.split('\n')
    cleaned_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
        elif in_code_block or line.strip().startswith('-') or line.strip().startswith('*'):
            # 代码块内或列表项，保留原样
            cleaned_lines.append(line)
        elif line.strip() == '':
            # 空行
            cleaned_lines.append('')
        else:
            # 普通行，移除行首多余空格
            cleaned_lines.append(line.lstrip())
    
    text = '\n'.join(cleaned_lines)
    
    # 清理多余的空白行（3个或更多换行符变成2个，但保留代码块周围的空行）
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'([^\n])\n{3,}([^\n])', r'\1\n\n\2', text)
    
    # 移除末尾的空白
    text = text.strip()
    
    return text

def create_slug(title):
    """根据标题创建URL友好的slug"""
    slug = title.lower()
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')
    return slug

def difficulty_to_chinese(difficulty):
    """将难度转换为中文"""
    difficulty_map = {
        "EASY": "Easy",
        "MEDIUM": "Medium",
        "HARD": "Hard"
    }
    return difficulty_map.get(difficulty, difficulty)

def format_challenge_markdown(challenge):
    """将题目数据格式化为Markdown"""
    title = challenge.get('title', 'Unknown')
    difficulty = difficulty_to_chinese(challenge.get('difficulty', 'Unknown'))
    spec_html = challenge.get('spec', '')
    spec_markdown = html_to_markdown(spec_html)
    challenge_id = challenge.get('id', '')
    
    md = f"# {title}\n\n"
    md += f"**难度**: {difficulty}\n\n"
    md += f"**题目ID**: {challenge_id}\n\n"
    
    # 构建URL
    slug = create_slug(title)
    if slug:
        md += f"**URL**: https://leetgpu.com/challenges/{slug}\n\n"
    else:
        md += f"**URL**: https://leetgpu.com/challenges/{challenge_id}\n\n"
    
    md += "---\n\n"
    
    if spec_markdown:
        md += spec_markdown
        md += "\n\n"
    
    md += "---\n\n"
    md += f"*最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    return challenge_id, slug, md

def main():
    """主函数：调用API获取所有题目并保存为Markdown文件"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 准备请求
    headers = {
        'accept': '*/*',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'authorization': AUTH_TOKEN,
        'content-type': 'application/json',
        'origin': 'https://leetgpu.com',
        'referer': 'https://leetgpu.com/',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    
    req = urllib.request.Request(API_URL, headers=headers)
    
    try:
        print("正在调用API获取题目列表...")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        print(f"成功获取 {len(data)} 个题目")
        
        success_count = 0
        failed_count = 0
        
        for challenge in data:
            try:
                challenge_id, slug, md_content = format_challenge_markdown(challenge)
                
                # 创建文件名: {id}_{slug}.md
                if slug:
                    filename = f"{challenge_id}_{slug}.md"
                else:
                    filename = f"{challenge_id}_challenge-{challenge_id}.md"
                
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                success_count += 1
                print(f"✓ 已保存: {filename}")
                
            except Exception as e:
                failed_count += 1
                print(f"✗ 保存失败 ({challenge.get('title', 'Unknown')}): {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n完成！成功: {success_count}, 失败: {failed_count}")
        print(f"文档保存在: {OUTPUT_DIR}/")
        
    except urllib.error.HTTPError as e:
        print(f"API调用失败: HTTP {e.code}")
        response_body = e.read().decode('utf-8')
        print(f"响应内容: {response_body}")
        print("\n提示：token可能已过期，请更新脚本中的 AUTH_TOKEN 变量")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

