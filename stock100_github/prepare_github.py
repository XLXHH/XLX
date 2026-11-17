"""Build a reviewed file allowlist, never include local secrets or cached data."""
import ast
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = ["app.py", "ai_defaults.py", "ai_prompt.py", "ai_evidence.py", "ai_news.py", "ai_report.py", "collector_v5.py",
         "pipeline.py", "research.py", "online.py", "events.py", "requirements.txt", "requirements-tested.txt",
         "start.ps1", "启动工具.cmd", "ai_local.py.example", ".gitignore", "test_ai_analysis.py", "GITHUB_UPLOAD.md", "prepare_github.py"]


def build():
    tree = ast.parse((ROOT / "collector_v5.py").read_text(encoding="utf-8-sig"))
    assignments = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EASTMONEY_COOKIE" for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0], ast.Call) or ast.unparse(assignments[0]) != "os.environ.get('EASTMONEY_COOKIE', '')":
        raise ValueError("collector_v5.py Cookie配置不符合空环境变量默认值，停止打包，请先核查")
    config = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
    config["v5_path"] = ""
    config["root"] = "research_data"
    target = ROOT.parent / "stock100_github.zip"
    readme = "# 股票100%\n\nStreamlit股票AI研判工具，使用DeepSeek-V4-Pro。\n\n安装与上传说明见 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。\n\n运行启动工具.cmd，抓取数据后在页面输入密钥。GitHub或Streamlit部署无需上传ai_local.py，入口为app.py。\n"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
        archive.writestr("training_config.json", json.dumps(config, ensure_ascii=False, indent=2))
        archive.writestr("README.md", readme)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert "ai_local.py" not in archive.namelist()
        assert all(not n.startswith(("logs/", "stock_cache/", "research_data/")) for n in archive.namelist())
    print(f"已生成 {target.name}；{len(FILES)+2}个文件，不含ai_local.py及本地数据。")


if __name__ == "__main__":
    build()
