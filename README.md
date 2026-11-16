# zh-sarcasm-lab

中文反讽与真实情感对比演示系统。  
本项目对比传统词频模型 `SnowNLP` 与轻量化上下文模型 `Erlangshen-RoBERTa-110M`（INT8 量化版）在正常评价与反讽吐槽上的表现差异，采用纯离线本地网页演示。

## 目录结构

```text
zh-sarcasm-lab/
├── README.md            # 项目说明与运行指南
├── config.json          # 全局配置：端口、模型路径、阈值、预设测试句
├── requirements.txt     # 运行依赖，无需 PyTorch
├── docs/
│   └── api.md           # 前后端接口约定
├── backend/             # 后端推理与本地 Web 服务
├── frontend/            # 网页界面交互
└── model/               # 离线模型目录，model_int8.onnx 需单独获取
```

## 环境准备

1. 安装依赖，建议先创建虚拟环境，Python 版本需 3.10 及以上：
   ```bash
   pip install -r requirements.txt
   ```
2. 放入离线模型文件：  
   `model_int8.onnx` 约 99 MB，未上传至 Git 仓库，请从 Releases 或团队共享群获取后放入 `model/` 目录：
   ```text
   model/
   ├── model_int8.onnx
   ├── tokenizer.json
   └── tokenizer_config.json
   ```

## 前后端接口

请先阅读 [docs/api.md](docs/api.md)，双方严格按约定的 JSON 格式并行开发，互不修改对方目录。

## 代码编写规范

1. **跨平台兼容**：
   - 文件读写显式指定 `encoding="utf-8"`，避免不同系统默认编码差异。
   - 禁止在代码中硬编码系统专属命令或路径分隔符 `\`。
   - Python 代码保持 3.10 及以上版本兼容。
2. **禁止写死路径与地址**：
   - 后端使用 `pathlib.Path(__file__).resolve().parent.parent` 定位项目根目录，再拼接相对路径。
   - 前端接口请求统一使用相对路径，如 `/api/analyze`，禁止写死 `http://127.0.0.1:8000`。
3. **纯离线约束**：
   - 前端禁止引入任何外网 CDN 链接、在线字体或外部在线脚本，所有页面资源本地化。
4. **配置外置与友好提示**：
   - 服务端口、模型文件名、最大长度、判定阈值及预设测试句统一从 [config.json](config.json) 读取。
   - 缺少模型文件等常见启动问题应输出清晰的中文提示。

## Git 协作规范

- `main` 分支仅保留可运行的稳定代码与公共文档，不直接强推未验证代码。
- 后端在 `feat/backend` 分支开发，前端在 `feat/frontend` 分支开发，各自只修改对应目录。
- 阶段功能完成后通过 Pull Request 合并回 `main`。
- 若修改了 [requirements.txt](requirements.txt)，提交后须在聊天群提醒队友，队友拉取最新代码后需重新执行 `pip install -r requirements.txt` 同步环境。
