import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pypdf import PdfWriter

from ai_evidence import build_ai_packet, summarize
from ai_news import enrich_latest, extract_pdf, official_pdf_url
from ai_prompt import SYSTEM_PROMPT, PROMPT_VERSION
from ai_report import split_report
from key_codec import encode_key, decode_key, KeyCodecError

REPORT_FIXTURE = """## 一、研判结论
等待价格确认，当前不追高。
## 二、证据与冲突
技术和资金尚未同向。
## 三、关键价位与转折
参考区间100～110元；这是条件假设。
## 四、行情路径
上涨需确认，跌破100元则推翻假设。
## 五、买卖计划
### 空仓者
等待确认后再评估。
1. 入场假设105元，失效100元。
2. 风险包括跳空，实际损失可能更大。
## 六、反证与风险
资金持续流出将削弱判断。
## 七、后续观察
观察收盘是否跌破100元。
"""


def packet_fixture():
    dates = pd.bdate_range("2026-06-01", periods=80)
    p = [{"date": d.isoformat(), "open": 100+i, "high": 102+i, "low": 98+i, "close": 100+i, "volume": 1000, "amount": 10000} for i, d in enumerate(dates)]
    f = [{"date": d.isoformat(), "main_net": 100} for d in dates[-30:]]
    n = [{"公告时间": f"2026-10-{i:02d}", "公告标题": f"测试公告{i}", "公告详情链接": f"https://www.cninfo.com.cn/{i}", "PDF下载链接": f"https://static.cninfo.com.cn/{i}.pdf"} for i in range(1, 9)]
    from financial_statements import collect_statements
    source = [{"SECURITY_CODE": "600519", "REPORT_DATE": "2026-09-30", "NOTICE_DATE": "2026-10-31", "TOTAL_OPERATE_INCOME": 300, "NETCASH_OPERATE": 100}]
    statements = collect_statements("600519", "2026-11-18", fetch=lambda code, report: source)
    return {"股票": "600519", "采集时间": "2026-11-17", "个股前复权日线": p, "沪深300日线": p, "东财主力净流入_元": f, "巨潮公告": n, "缺失与限制": [], "公司经营数据": {"报表金额补充": statements}}


class AnalysisTests(unittest.TestCase):
    def test_key_codec_roundtrip_shortening_and_rejection(self):
        keys = ["sk-" + "0123456789abcdef"*2, "sk-" + "0"*64, "sk-"+"a"*48, "sk-AbC_123-xyz", "test-key"]
        for key in keys:
            self.assertEqual(decode_key(encode_key(key)), key)
        self.assertLess(len(encode_key(keys[0])), len(keys[0]))
        for bad in ["sk-raw-key", "", "D1.!", "K1.A", "K1.Cgo", "D1." + "A"*1100]:
            with self.assertRaises(KeyCodecError):
                decode_key(bad)

    def test_cloud_without_local_config_and_page_key(self):
        # Streamlit unloads application imports; isolate the alternate config import
        # so Plotly class caches cannot cross between two distinct app instances.
        import os
        import subprocess
        import sys
        if os.environ.get("STOCK_CLOUD_TEST_CHILD") != "1":
            run = subprocess.run([sys.executable, "-m", "unittest", "test_ai_analysis.AnalysisTests.test_cloud_without_local_config_and_page_key"],
                                 cwd=Path(__file__).parent, env={**os.environ, "STOCK_CLOUD_TEST_CHILD": "1"}, capture_output=True, timeout=50)
            self.assertEqual(run.returncode, 0, run.stderr.decode("utf-8", errors="replace"))
            return
        from streamlit.testing.v1 import AppTest
        response = SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": REPORT_FIXTURE}}]})
        with patch.dict("sys.modules", {"ai_local": None}), patch("requests.post", return_value=response) as post, patch("ai_requests.AnalysisRequests.audit"):
            app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py"))).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(any(b.label == "抓取最新资料" for b in app.button))
            next(t for t in app.text_input if t.label == "访问密码").input("wrong").run(timeout=30)
            next(b for b in app.button if b.label == "进入看板").click().run(timeout=30)
            self.assertTrue(any("密码不正确" in e.value for e in app.error))
            self.assertEqual(post.call_count, 0)
            next(t for t in app.text_input if t.label == "访问密码").input("xlx222").run(timeout=30)
            next(b for b in app.button if b.label == "进入看板").click().run(timeout=30)
            self.assertTrue(app.session_state["site_authenticated"])
            app.session_state["packet"] = packet_fixture()
            app.run(timeout=30)
            next(b for b in app.button if b.label == "发送以上资料并分析").click().run(timeout=30)
            self.assertEqual(post.call_count, 0)
            self.assertTrue(any("填写编码后的 DeepSeek 密钥" in error.value for error in app.error))
            next(t for t in app.text_input if t.label == "编码后的 DeepSeek 密钥").input("sk-raw-key").run(timeout=30)
            next(b for b in app.button if b.label == "发送以上资料并分析").click().run(timeout=30)
            self.assertEqual(post.call_count, 0)
            encoded = encode_key("test-session-only-key")
            next(t for t in app.text_input if t.label == "编码后的 DeepSeek 密钥").input(encoded).run(timeout=30)
            next(b for b in app.button if b.label == "发送以上资料并分析").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-session-only-key")
            self.assertNotIn("test-session-only-key", json.dumps(post.call_args.kwargs["json"]))
            self.assertNotIn("test-session-only-key", json.dumps(app.session_state["packet"]))
            self.assertNotIn(encoded, json.dumps(post.call_args.kwargs["json"]))
            next(b for b in app.button if b.label == "发送以上资料并分析").click().run(timeout=30)
            self.assertEqual(post.call_count, 1)
            self.assertTrue(any("没有再次调用AI" in info.value for info in app.info))
            next(b for b in app.button if b.label == "退出登录").click().run(timeout=30)
            self.assertFalse("site_authenticated" in app.session_state)
            self.assertFalse("analysis_api_key" in app.session_state)
            self.assertFalse("packet" in app.session_state)
            self.assertTrue(any(t.label == "访问密码" for t in app.text_input))

    def test_report_layout_preserves_nested_risk_and_fallback(self):
        blocks = split_report(REPORT_FIXTURE)
        self.assertEqual(set(blocks), set(range(1, 8)))
        self.assertIn("失效100元", blocks[5])
        self.assertIn("实际损失可能更大", blocks[5])
        self.assertIn("资金持续流出", blocks[6])
        plain = "格式不符合要求，但必须保留：风险可能超过5%。"
        self.assertEqual(split_report(plain), {0: plain})
        fenced = "```text\n## 六、反证与风险\n完整原文\n```"
        self.assertEqual(split_report(fenced), {0: fenced})

    def test_calculation_and_missing_flow(self):
        p = packet_fixture()
        result = summarize(p)
        self.assertEqual(result["行情"]["均线"]["MA5"]["数值"], 177)
        self.assertEqual(result["行情"]["候选结构价位_不是自动买卖信号"]["前5条_不含当日"]["最高价"], 180)
        self.assertEqual(result["行情"]["ATR14_简单平均真实波幅"], 4)
        self.assertEqual(result["资金"]["与行情同日窗口"]["最近5条行情记录"]["净流入合计"], 500)
        self.assertEqual(result["资金"]["与行情同日窗口"]["最近5条行情记录"]["净流入合计除成交额合计"], .01)
        p["东财主力净流入_元"].pop(-2)
        missing = summarize(p)["资金"]["与行情同日窗口"]["最近5条行情记录"]
        self.assertFalse(missing["完整覆盖"])
        self.assertIsNone(missing["净流入合计"])

    def test_future_data_does_not_enter_prior_extremes_or_flow_window(self):
        p = packet_fixture()
        p["个股前复权日线"][-1]["high"] = 10000
        p["东财主力净流入_元"].append({"date": "2099-01-01", "main_net": 1000000})
        r = summarize(p)
        self.assertEqual(r["行情"]["候选结构价位_不是自动买卖信号"]["前5条_不含当日"]["最高价"], 180)
        self.assertEqual(r["资金"]["与行情同日窗口"]["最近1条行情记录"]["净流入合计"], 100)

    def test_five_latest_text_matching_no_mutation(self):
        p = packet_fixture()
        original = json.dumps(p)
        text = {**p["巨潮公告"][-1], "正文状态": "已提取正文", "正文摘录": "真实摘录"}
        stale = {**p["巨潮公告"][-2], "公告时间": "1999-01-01", "正文状态": "已提取正文", "正文摘录": "错配内容"}
        p["巨潮公告正文"] = [text, stale]
        data = build_ai_packet(p, [text, stale])
        self.assertNotIn("巨潮公告正文", data)
        self.assertEqual(len(data["巨潮公告"]), 5)
        self.assertEqual(data["巨潮公告"][0]["公告标题"], "测试公告8")
        self.assertEqual(data["巨潮公告"][0]["正文摘录"], "真实摘录")
        self.assertEqual(data["巨潮公告"][1]["正文状态"], "未读取正文")
        self.assertNotIn("错配内容", json.dumps(data, ensure_ascii=False))
        del p["巨潮公告正文"]
        self.assertEqual(json.dumps(p), original)
        json.dumps(data, ensure_ascii=False, allow_nan=False)

    def test_pdf_failure_cannot_be_claimed_read(self):
        def bad_fetch(url):
            raise TimeoutError()
        rows = enrich_latest(packet_fixture()["巨潮公告"], fetch=bad_fetch)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(x["正文状态"] == "未读取正文" and "正文摘录" not in x for x in rows))
        for url in ["https://evil.example/a.pdf", "https://static.cninfo.com.cn.evil.example/a.pdf", "http://localhost/a.pdf"]:
            with self.assertRaises(ValueError):
                official_pdf_url(url)
        self.assertEqual(official_pdf_url("http://static.cninfo.com.cn/a.pdf"), "https://static.cninfo.com.cn/a.pdf")

    def test_pdf_excerpt_pages_and_scanned(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        b = io.BytesIO()
        writer.write(b)
        with self.assertRaises(ValueError):
            extract_pdf(b.getvalue())
        pages = [SimpleNamespace(extract_text=lambda i=i: f"公告正文第{i+1}页" + "文字"*400) for i in range(24)]
        with patch("ai_news.PdfReader", return_value=SimpleNamespace(pages=pages)):
            out = extract_pdf(b"%PDF-fake")
        self.assertEqual(out["正文状态"], "部分正文")
        self.assertLessEqual(len(out["正文摘录"]), 8000)
        self.assertIn("[第24页]", out["正文摘录"])
        self.assertNotIn(17, out["提取到文字的页码"])

    def test_app_passes_prompt_and_selected_evidence_without_real_api(self):
        from streamlit.testing.v1 import AppTest
        request_calls = []
        def reply(url, **kwargs):
            request_calls.append(kwargs)
            return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": REPORT_FIXTURE}}]})
        with patch.dict("sys.modules", {"ai_local": None}), patch("ai_defaults.BASE_URL", "https://api.deepseek.com"), patch("ai_defaults.MODEL", "deepseek-flash"), patch("ai_defaults.API_KEY", "test-key"), patch("requests.post", side_effect=reply), patch("ai_requests.AnalysisRequests.audit"):
            app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py"))).run(timeout=30)
            app.session_state["site_authenticated"] = True
            app.session_state["packet"] = packet_fixture()
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            next(b for b in app.button if b.label == "发送以上资料并分析").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            sent = request_calls[0]["json"]["messages"]
            self.assertEqual(request_calls[0]["json"]["model"], "deepseek-flash")
            self.assertEqual(request_calls[0]["json"]["thinking"], {"type": "enabled"})
            self.assertEqual(request_calls[0]["json"]["reasoning_effort"], "high")
            self.assertEqual(request_calls[0]["json"]["max_tokens"], 32768)
            self.assertEqual(request_calls[0]["timeout"], (10, 300))
            self.assertEqual(sent[0]["content"], SYSTEM_PROMPT)
            data = json.loads(sent[1]["content"])
            self.assertEqual(len(data["巨潮公告"]), 5)
            self.assertIn("本地计算摘要", data)
            self.assertEqual(data["公司经营数据"]["报表金额补充"]["最近四个报告期"][0]["利润表"]["累计金额"]["营业总收入"], 300)
            self.assertEqual(data["分析设定"]["提示词版本"], PROMPT_VERSION)
            self.assertEqual(app.session_state["answer_version"], PROMPT_VERSION)
            self.assertTrue({"证据与价位", "行情路径", "买卖计划", "风险与跟踪"}.issubset({t.label for t in app.tabs}))
            self.assertTrue(any("实际损失可能更大" in md.value for md in app.markdown))


if __name__ == "__main__":
    unittest.main()
