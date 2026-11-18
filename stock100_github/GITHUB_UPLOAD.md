# GitHub 上传与本地启动

上传包已按明确文件清单生成，仅包含代码、依赖、启动脚本、说明和空白配置模板。解压 stock100_github.zip 后，将里面文件放到仓库根目录。GitHub用于保存代码；仅上传不会自动启动Streamlit。

## 新电脑运行

1. 安装Python 3.11或更新版本。
2. 运行 `编码密钥.cmd`，输入原始DeepSeek API Key，复制生成的编码。访问看板先验证密码，抓取资料后在发送按钮上方填写编码。也可将 `ai_local.py.example` 复制成 `ai_local.py`，在本机配置原始密钥作为备用。
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

- streamlit_app.py：浅蓝看板及AI调用。
- access_control.py、key_codec.py、编码密钥.cmd：访问验证、密钥还原与本机编码工具。
- ai_prompt.py、ai_evidence.py、ai_news.py、ai_report.py：分析提示词、确定性摘要、PDF正文及报告展示。
- collector_v5.py、pipeline.py、research.py、online.py、events.py：采集与现有代码依赖。
- requirements.txt、start.ps1、启动工具.cmd：安装与本地运行。
- training_config.json、ai_local.py.example、.gitignore：配置与排除规则。
- test_ai_analysis.py：离线功能及模拟AI页面测试，不调用付费服务。
- ai_requests.py、test_ai_requests.py：同批资料重复提交拦截、请求编号和tokens记录，超时不自动重试。

若未来部署到Streamlit Community Cloud，需要单独配置云端Secrets及网络可达性；本上传包不代表已经完成云端部署。

使用GitHub仓库部署Streamlit时，入口选streamlit_app.py。缺少ai_local.py会自动使用公开ai_defaults.py中的模型设置；访客通过访问验证后，在页面填写编码后的密钥，不需要上传真实密钥文件。东方财富如需Cookie，可通过云端环境配置，仍需验证行情/公告接口在云端的可达性。页面密钥不会写入来源JSON或下载报告。编码结果同样不要上传，详见KEY_ENCODING.md。

## 请求对账与重复提交

默认使用 deepseek-flash 和 high 思考强度。报告篇幅由证据和策略复杂度决定。每次最多生成 32768 tokens，包含思考和最终正文，按实际消耗计费。若本机存在 ai_local.py，以其中设置为准。

同一浏览器会话、服务器未重启期间，同批资料重复点击会复用已完成结果；正在执行时不重复发送，超时或异常后也不自动重试。改变页面筛选不会自动调用AI。重新采集、换浏览器会话、重启服务后可能产生新请求，不能跨服务器保证幂等。

每次调用在“本次调用与费用记录”显示本地编号、服务返回编号、时间、状态、输入/输出/总tokens，可下载该次JSON用于对账。费用是按2026-11-18官方价格估算的空闲/高峰范围，并非实际扣款通知。接口未返回用量时不编造数值。

服务器logs/ai_requests.jsonl只保存请求元数据，不含密钥、编码、提示词、行情或公告正文。一条HTTP请求会有开始和结束两条事件，它们的local_request_id相同，不是两次计费。Streamlit云端磁盘与进程内的重复拦截记录可能在重启时丢失，账单明细仍以DeepSeek后台为准。更新前的历史请求无法从新日志补回。
