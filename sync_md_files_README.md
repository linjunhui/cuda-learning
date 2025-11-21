# MD文件同步脚本使用说明

## 功能说明

`sync_md_files.sh` 脚本用于将当前目录及所有子目录下的 `.md` 文件同步到 `文档收集` 目录，保持原有的目录结构。

## 脚本特性

- ✅ 自动查找所有 `.md` 文件
- ✅ 保持原有目录结构
- ✅ 增量同步（只同步更新的文件）
- ✅ 详细的日志记录
- ✅ 统计同步结果

## 使用方法

### 1. 手动执行

```bash
# 在项目根目录执行
./sync_md_files.sh
```

### 2. 定期自动执行（使用 cron）

#### 编辑 crontab

```bash
crontab -e
```

#### 添加定时任务

以下是一些常用的定时配置示例：

```bash
# 每 30 分钟执行一次
*/30 * * * * cd /home/jonson/cuda-learning && /home/jonson/cuda-learning/sync_md_files.sh

# 每小时执行一次
0 * * * * cd /home/jonson/cuda-learning && /home/jonson/cuda-learning/sync_md_files.sh

# 每天凌晨 2 点执行一次
0 2 * * * cd /home/jonson/cuda-learning && /home/jonson/cuda-learning/sync_md_files.sh

# 每 6 小时执行一次
0 */6 * * * cd /home/jonson/cuda-learning && /home/jonson/cuda-learning/sync_md_files.sh
```

**注意**：请将路径替换为你的实际项目路径。

### 3. 使用 systemd timer（推荐，适用于 systemd 系统）

#### 创建 service 文件

创建 `/etc/systemd/system/sync-md-files.service`：

```ini
[Unit]
Description=Sync MD files to 文档收集 directory
After=network.target

[Service]
Type=oneshot
User=jonson
WorkingDirectory=/home/jonson/cuda-learning
ExecStart=/home/jonson/cuda-learning/sync_md_files.sh
```

#### 创建 timer 文件

创建 `/etc/systemd/system/sync-md-files.timer`：

```ini
[Unit]
Description=Timer for syncing MD files
Requires=sync-md-files.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=30min
Unit=sync-md-files.service

[Install]
WantedBy=timers.target
```

#### 启用和启动 timer

```bash
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 启用 timer（开机自启）
sudo systemctl enable sync-md-files.timer

# 启动 timer
sudo systemctl start sync-md-files.timer

# 查看 timer 状态
sudo systemctl status sync-md-files.timer

# 查看下次执行时间
sudo systemctl list-timers sync-md-files.timer
```

### 4. 使用 while 循环（简单但占用资源）

如果需要简单的定期执行，可以在后台运行：

```bash
# 后台运行，每 30 分钟执行一次
while true; do
    /home/jonson/cuda-learning/sync_md_files.sh
    sleep 1800  # 1800 秒 = 30 分钟
done &
```

## 日志文件

脚本执行日志保存在 `sync_md_files.log` 文件中，包含：
- 执行时间
- 同步的文件列表
- 统计信息（同步数、跳过数、错误数）

查看日志：

```bash
# 查看最新日志
tail -f sync_md_files.log

# 查看最近 50 行
tail -n 50 sync_md_files.log
```

## 配置说明

脚本中的主要配置变量：

- `SOURCE_DIR`: 源目录（默认为脚本所在目录）
- `TARGET_DIR`: 目标目录（默认为 `文档收集`）
- `LOG_FILE`: 日志文件路径
- `DATE_FORMAT`: 日志时间格式

可以根据需要修改这些变量。

## 注意事项

1. **路径问题**：确保脚本路径和目标路径正确
2. **权限问题**：确保脚本有执行权限（`chmod +x sync_md_files.sh`）
3. **磁盘空间**：定期检查 `文档收集` 目录的磁盘使用情况
4. **排除目录**：脚本会自动排除 `文档收集` 目录本身，避免递归复制

## 测试

首次使用前，建议先手动执行一次，检查同步结果：

```bash
# 执行脚本
./sync_md_files.sh

# 查看日志
cat sync_md_files.log

# 检查同步结果
ls -R 文档收集/
```

## 故障排查

如果同步失败，检查：

1. 脚本是否有执行权限：`ls -l sync_md_files.sh`
2. 目标目录是否可写：`touch 文档收集/test && rm 文档收集/test`
3. 查看日志文件：`cat sync_md_files.log`
4. 手动执行脚本查看错误信息

## 示例输出

```
[2024-01-15 10:30:00] ========== 开始同步 MD 文件 ==========
[2024-01-15 10:30:00] 源目录: /home/jonson/cuda-learning
[2024-01-15 10:30:00] 目标目录: /home/jonson/cuda-learning/文档收集
[2024-01-15 10:30:01] ✓ 同步: README.md
[2024-01-15 10:30:01] ✓ 同步: C++学习/README.md
[2024-01-15 10:30:01] ✓ 同步: CUDA学习/README.md
[2024-01-15 10:30:02] ========== 同步完成 ==========
[2024-01-15 10:30:02] 同步文件数: 150
[2024-01-15 10:30:02] 跳过文件数: 20
[2024-01-15 10:30:02] 错误文件数: 0
```
