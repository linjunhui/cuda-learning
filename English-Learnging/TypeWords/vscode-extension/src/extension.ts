import * as vscode from 'vscode';
import axios from 'axios';

let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
    console.log('TypeWords 插件已激活');

    // 创建状态栏项
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = "$(book) TypeWords";
    statusBarItem.tooltip = "点击打开 TypeWords";
    statusBarItem.command = 'typewords.open';
    statusBarItem.show();

    // 注册命令：打开 TypeWords
    const openCommand = vscode.commands.registerCommand('typewords.open', () => {
        const config = vscode.workspace.getConfiguration('typewords');
        const serverUrl = config.get<string>('serverUrl', 'http://47.108.93.168:48372');
        
        vscode.env.openExternal(vscode.Uri.parse(serverUrl));
        vscode.window.showInformationMessage('正在打开 TypeWords...');
    });

    // 注册命令：练习单词
    const practiceWordCommand = vscode.commands.registerCommand('typewords.practiceWord', async () => {
        const config = vscode.workspace.getConfiguration('typewords');
        const serverUrl = config.get<string>('serverUrl', 'http://47.108.93.168:48372');
        
        try {
            const url = `${serverUrl}/#/word`;
            await vscode.env.openExternal(vscode.Uri.parse(url));
            vscode.window.showInformationMessage('正在打开单词练习...');
        } catch (error) {
            vscode.window.showErrorMessage(`打开失败: ${error}`);
        }
    });

    // 注册命令：练习文章
    const practiceArticleCommand = vscode.commands.registerCommand('typewords.practiceArticle', async () => {
        const config = vscode.workspace.getConfiguration('typewords');
        const serverUrl = config.get<string>('serverUrl', 'http://47.108.93.168:48372');
        
        try {
            const url = `${serverUrl}/#/article`;
            await vscode.env.openExternal(vscode.Uri.parse(url));
            vscode.window.showInformationMessage('正在打开文章练习...');
        } catch (error) {
            vscode.window.showErrorMessage(`打开失败: ${error}`);
        }
    });

    // 注册命令：翻译选中文本
    const translateCommand = vscode.commands.registerCommand('typewords.translate', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('请先打开一个文件');
            return;
        }

        const selection = editor.selection;
        const text = editor.document.getText(selection);
        
        if (!text) {
            vscode.window.showWarningMessage('请先选中要翻译的文本');
            return;
        }

        try {
            const config = vscode.workspace.getConfiguration('typewords');
            const serverUrl = config.get<string>('serverUrl', 'http://47.108.93.168:48372');
            
            // 打开翻译页面，可以通过 URL 参数传递文本
            const url = `${serverUrl}/#/translate?text=${encodeURIComponent(text)}`;
            await vscode.env.openExternal(vscode.Uri.parse(url));
            vscode.window.showInformationMessage('正在打开翻译页面...');
        } catch (error) {
            vscode.window.showErrorMessage(`翻译失败: ${error}`);
        }
    });

    // 将所有命令添加到订阅中
    context.subscriptions.push(
        statusBarItem,
        openCommand,
        practiceWordCommand,
        practiceArticleCommand,
        translateCommand
    );

    // 如果配置了自动打开，则在启动时打开
    const config = vscode.workspace.getConfiguration('typewords');
    const autoOpen = config.get<boolean>('autoOpen', false);
    if (autoOpen) {
        setTimeout(() => {
            vscode.commands.executeCommand('typewords.open');
        }, 2000);
    }
}

export function deactivate() {
    if (statusBarItem) {
        statusBarItem.dispose();
    }
}
