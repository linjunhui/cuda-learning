# Gitee仓库创建和推送指南

## 概述
本指南将详细说明如何在Gitee（码云）平台上创建新仓库，并将当前目录的CUDA学习项目推送到远程仓库。

## 前提条件
- 已注册Gitee账号：https://gitee.com/linjunhui
- 已安装Git工具
- 当前项目目录：`/home/jonson/cuda-learning`

## 步骤一：在Gitee上创建新仓库

### 1.1 登录Gitee
1. 打开浏览器，访问 https://gitee.com
2. 点击右上角"登录"，使用您的账号登录

### 1.2 创建新仓库
1. 登录后，点击右上角的"+"号，选择"新建仓库"
2. 填写仓库信息：
   - **仓库名称**：`cuda-learning`（建议与本地目录名一致）
   - **仓库描述**：`CUDA编程学习项目，包含各种练习和示例代码`
   - **仓库类型**：选择"公开"或"私有"（根据您的需要）
   - **是否开源**：选择"是"（如果选择公开）
   - **添加 .gitignore**：选择"None"（因为我们要推送现有代码）
   - **添加开源许可证**：可选择"MIT License"或其他合适的许可证
3. 点击"创建"按钮

### 1.3 获取仓库地址
创建完成后，您会看到仓库的详细信息页面。复制仓库的HTTPS地址，格式类似：
```
https://gitee.com/linjunhui/cuda-learning.git
```

## 步骤二：初始化本地Git仓库

### 2.1 打开终端
在您的项目目录中打开终端（Linux环境下使用WSL2）

### 2.2 初始化Git仓库
```bash
cd /home/jonson/cuda-learning
git init
```

### 2.3 添加远程仓库
```bash
git remote add origin https://gitee.com/linjunhui/cuda-learning.git
```

## 步骤三：配置Git用户信息（如果未配置）

### 3.1 设置用户名和邮箱
```bash
git config --global user.name "您的用户名"
git config --global user.email "您的邮箱@example.com"
```

### 3.2 验证配置
```bash
git config --list
```

## 步骤四：准备推送文件

### 4.1 创建 .gitignore 文件
为了避免推送不必要的文件，建议创建 `.gitignore` 文件：

```bash
cat > .gitignore << 'EOF'
# 编译生成的文件
*.o
*.obj
*.exe
*.out
*.bin

# 调试文件
*.pdb
*.ilk

# IDE相关文件
.vscode/
.idea/
*.swp
*.swo
*~

# 系统文件
.DS_Store
Thumbs.db

# 临时文件
*.tmp
*.temp
*.log

# CUDA编译中间文件
*.cubin
*.ptx
*.fatbin
EOF
```

### 4.2 添加所有文件到暂存区
```bash
git add .
```

### 4.3 提交更改
```bash
git commit -m "Initial commit: CUDA学习项目初始化

- 添加CUDA编程学习材料
- 包含chapter2-cuda-programming模块
- 包含CUDA每日一练
- 包含C++每日一练
- 包含LeetCode每日一练
- 包含leetgpu项目
- 添加配置文件"
```

## 步骤五：推送到Gitee

### 5.1 推送到主分支
```bash
git push -u origin master
```

**注意**：如果Gitee默认分支是 `main` 而不是 `master`，请使用：
```bash
git push -u origin main
```

### 5.2 如果遇到认证问题
如果推送时需要输入用户名和密码：
- **用户名**：您的Gitee用户名
- **密码**：您的Gitee密码（或访问令牌）

## 步骤六：验证推送结果

### 6.1 检查远程仓库
访问您的Gitee仓库页面：
```
https://gitee.com/linjunhui/cuda-learning
```

确认所有文件都已成功上传。

### 6.2 检查本地状态
```bash
git status
```

应该显示 "nothing to commit, working tree clean"。

## 常见问题解决

### 问题1：推送被拒绝
**错误信息**：`rejected: master -> master (fetch first)`

**解决方案**：
```bash
git pull origin master --allow-unrelated-histories
git push -u origin master
```

### 问题2：认证失败
**解决方案**：
1. 检查用户名和密码是否正确
2. 考虑使用访问令牌替代密码
3. 在Gitee设置中生成个人访问令牌

### 问题3：分支名称不匹配
**解决方案**：
```bash
# 查看远程分支
git branch -r

# 如果远程是main分支，本地创建并切换
git checkout -b main
git push -u origin main
```

## 后续操作建议

### 1. 创建分支进行开发
```bash
git checkout -b develop
# 进行开发工作
git add .
git commit -m "开发功能描述"
git push -u origin develop
```

### 2. 定期同步
```bash
git pull origin master  # 拉取最新更改
git push origin master  # 推送本地更改
```

### 3. 添加协作成员
在Gitee仓库页面：
1. 点击"管理"
2. 选择"仓库成员管理"
3. 添加协作者

## 项目结构说明

当前项目包含以下主要模块：
- `chapter2-cuda-programming/` - CUDA编程第二章内容
- `CUDA每日一练/` - CUDA相关练习题
- `Cplusplus每日一练/` - C++相关练习题
- `LeetCode每日一练/` - LeetCode算法题练习
- `leetgpu/` - GPU相关的LeetCode解决方案
- `config/` - 配置文件
- `README.md` - 项目说明文档

## 总结

通过以上步骤，您已经成功：
1. 在Gitee上创建了新的仓库
2. 初始化了本地Git仓库
3. 配置了远程仓库连接
4. 推送了所有项目文件到远程仓库

现在您可以在Gitee上查看您的CUDA学习项目，并与其他人分享或协作开发。

---

**注意事项**：
- 请确保不要推送包含敏感信息的文件
- 定期备份重要代码
- 使用有意义的提交信息
- 遵循良好的版本控制实践
