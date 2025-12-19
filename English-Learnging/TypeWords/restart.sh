#!/bin/bash

# 定义要查询的端口
PORT=48372

# 方法1：使用 lsof 查询端口进程（推荐，大部分系统自带）
PID=$(lsof -ti:$PORT)

# 方法2：如果系统没有 lsof，可使用 netstat（需安装 net-tools）
# PID=$(netstat -nlp | grep :$PORT | awk '{print $7}' | cut -d '/' -f 1)

# 判断是否找到进程
if [ -n "$PID" ]; then
    echo "找到占用端口 $PORT 的进程，PID：$PID"
    # 终止进程（-9 是强制终止，可根据需要改为 -15 优雅终止）
    kill -9 $PID
    if [ $? -eq 0 ]; then
        echo "进程 $PID 已成功终止"
    else
        echo "终止进程 $PID 失败"
        exit 1
    fi
else
    echo "端口 $PORT 未被占用"
fi

# 执行 npm run serve
echo "开始执行 npm run serve..."
#rm -rf dist
npm run build
nohup npm run serve > serve.log 2>&1 &

