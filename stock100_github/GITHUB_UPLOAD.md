# GitHub 上传与本地启动

上传包已按明确文件清单生成，仅包含代码、依赖、启动脚本、说明和空白配置模板。解压 stock100_github.zip 后，将里面文件放到仓库根目录。GitHub用于保存代码；仅上传不会自动启动Streamlit。

## 新电脑运行

1. 安装Python 3.11或更新版本。
2. 抓取资料后在页面“发送以上资料并分析”上方填写DeepSeek API Key。也可将 `ai_local.py.example` 复制成 `ai_local.py`，在本机配置密钥作为备用。
3. 东方财富Cookie通过本机环境变量 `EASTMONEY_COOKIE` 提供；或在本机 `training_config.json` 的v5_path中填写现有v5脚本路径。上传包中的路径为空。
4. 双击 `启动工具.cmd`，安装依赖后启动本地页面。

当前程序只在线采集并调用AI；pipeline.py、research.py、online.py等仍是导入依赖，因此需要保留，但不需要执行股票池训练。

## 不要上传

- `ai_local.py`：真实密钥配置。
- `.streamlit/secrets.toml`、`.env`：真实凭据。
- 你D盘原始v5文件：其中可能有Cookie。上传包使用项目内不含硬编码Cookie的collector_v5.py。
- 虚拟环境、缓存、公告文件、日志、research_data、模型与下载结果。

`.gitignore`只对Git提交时尚未跟踪的文件生效，不会替你撤销历史提交，也不会阻止手动网页上传选错文件。若密钥曾上传，应在服务商处撤销并更换。

## 当前代码

- app.py：浅蓝看板及AI调用。
- ai_prompt.py、ai_evidence.py、ai_news.py、ai_report.py：分析提示词、确定性摘要、PDF正文及报告展示。
- collector_v5.py、pipeline.py、research.py、online.py、events.py：采集与现有代码依赖。
- requirements.txt、start.ps1、启动工具.cmd：安装与本地运行。
- training_config.json、ai_local.py.example、.gitignore：配置与排除规则。
- test_ai_analysis.py：离线功能及模拟AI页面测试，不调用付费服务。

若未来部署到Streamlit Community Cloud，需要单独配置云端Secrets及网络可达性；本上传包不代表已经完成云端部署。

使用GitHub仓库部署Streamlit时，入口选app.py。缺少ai_local.py会自动使用公开ai_defaults.py中的模型设置；访客可在页面密码框填写密钥，不需要上传真实密钥文件。东方财富如需Cookie，可通过云端环境配置，仍需验证行情/公告接口在云端的可达性。页面密钥不会写入来源JSON或下载报告。
