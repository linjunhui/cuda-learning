#!/bin/bash

# MD文件同步脚本
# 功能：将当前目录及子目录下的所有 .md 文件同步到 文档收集 目录
# 使用方法：
#   1. 手动执行：./sync_md_files.sh
#   2. 定期执行：使用 cron 或 systemd timer

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR"
TARGET_DIR="$SCRIPT_DIR/文档收集"
LOG_FILE="$SCRIPT_DIR/sync_md_files.log"
DATE_FORMAT="%Y-%m-%d %H:%M:%S"

# 创建日志函数
log() {
    echo "[$(date +"$DATE_FORMAT")] $1" | tee -a "$LOG_FILE"
}

# 检查目标目录是否存在，不存在则创建
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
    log "创建目标目录: $TARGET_DIR"
fi

# 开始同步
log "========== 开始同步 MD 文件 =========="
log "源目录: $SOURCE_DIR"
log "目标目录: $TARGET_DIR"

# 计数器
SYNCED_COUNT=0
SKIPPED_COUNT=0
ERROR_COUNT=0

# 查找所有 .md 文件并同步
# 排除目标目录本身，避免递归复制
# 使用进程替换避免子 shell 问题
while IFS= read -r file; do
    # 计算相对路径
    rel_path="${file#$SOURCE_DIR/}"
    
    # 构建目标文件路径
    target_file="$TARGET_DIR/$rel_path"
    target_dir="$(dirname "$target_file")"
    
    # 创建目标目录（如果不存在）
    if [ ! -d "$target_dir" ]; then
        mkdir -p "$target_dir"
    fi
    
    # 检查是否需要同步（文件不存在或源文件更新）
    if [ ! -f "$target_file" ] || [ "$file" -nt "$target_file" ]; then
        # 复制文件
        if cp -f "$file" "$target_file" 2>/dev/null; then
            ((SYNCED_COUNT++))
            log "✓ 同步: $rel_path"
        else
            ((ERROR_COUNT++))
            log "✗ 错误: 无法同步 $rel_path"
        fi
    else
        ((SKIPPED_COUNT++))
    fi
done < <(find "$SOURCE_DIR" -type f -name "*.md" ! -path "$TARGET_DIR/*")

# 统计结果
log "========== 同步完成 =========="
log "同步文件数: $SYNCED_COUNT"
log "跳过文件数: $SKIPPED_COUNT"
log "错误文件数: $ERROR_COUNT"
log ""

# 清理目标目录中已不存在的源文件（可选，谨慎使用）
# 如果需要启用此功能，取消下面的注释
# log "清理目标目录中已删除的源文件..."
# find "$TARGET_DIR" -type f -name "*.md" | while read -r target_file; do
#     rel_path="${target_file#$TARGET_DIR/}"
#     source_file="$SOURCE_DIR/$rel_path"
#     if [ ! -f "$source_file" ]; then
#         rm -f "$target_file"
#         log "删除: $rel_path (源文件已不存在)"
#     fi
# done

exit 0
