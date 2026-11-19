import json
import unittest
from unittest.mock import patch

import requests

from financial_statements import collect_statements, display_rows, fetch_rows


def rows():
    return [
        {"SECURITY_CODE": "600519", "REPORT_DATE": "2026-09-30", "NOTICE_DATE": "2026-10-31", "TOTAL_OPERATE_INCOME": 300, "OPERATE_INCOME": None, "NETCASH_OPERATE": -5},
        {"SECURITY_CODE": "600519", "REPORT_DATE": "2026-06-30", "NOTICE_DATE": "2026-08-15", "TOTAL_OPERATE_INCOME": 180, "NETCASH_OPERATE": 10},
        {"SECURITY_CODE": "600519", "REPORT_DATE": "2026-03-31", "NOTICE_DATE": "2026-04-25", "TOTAL_OPERATE_INCOME": 80, "NETCASH_OPERATE": 0},
        {"SECURITY_CODE": "600519", "REPORT_DATE": "2025-09-30", "NOTICE_DATE": "2025-10-31", "TOTAL_OPERATE_INCOME": 200, "NETCASH_OPERATE": 20},
    ]


class StatementTests(unittest.TestCase):
    def test_cumulative_quarterly_yoy_zero_and_negative_cash(self):
        result = collect_statements("sh.600519", "2026-11-18", fetch=lambda code, report: rows())
        current = result["最近四个报告期"][0]
        income = current["利润表"]
        self.assertEqual(income["累计金额"]["营业总收入"], 300)
        self.assertIsNone(income["累计金额"]["营业收入"])
        self.assertIn("营业收入", income["缺失字段"])
        self.assertEqual(income["单季度推导金额"]["营业总收入"], 120)
        self.assertEqual(income["累计同比增长率_小数"]["营业总收入"], .5)
        self.assertEqual(current["现金流量表"]["累计金额"]["经营活动现金流量净额"], -5)
        self.assertEqual(current["现金流量表"]["单季度推导金额"]["经营活动现金流量净额"], -15)
        q1 = next(row for row in result["最近四个报告期"] if row["季度"] == "2026Q1")
        self.assertEqual(q1["现金流量表"]["累计金额"]["经营活动现金流量净额"], 0)
        self.assertEqual(q1["现金流量表"]["单季度推导金额"]["经营活动现金流量净额"], 0)
        self.assertEqual(display_rows(result)[0]["营业总收入（亿元）"], 300/1e8)
        json.dumps(result, allow_nan=False)

    def test_future_notices_wrong_stock_and_missing_dates_are_excluded(self):
        source = rows()+[
            {"SECURITY_CODE": "600519", "REPORT_DATE": "2026-12-31", "NOTICE_DATE": "2027-04-01", "TOTAL_OPERATE_INCOME": 9999},
            {"SECURITY_CODE": "600000", "REPORT_DATE": "2026-09-30", "NOTICE_DATE": "2026-10-30", "TOTAL_OPERATE_INCOME": 8888},
            {"SECURITY_CODE": "600519", "REPORT_DATE": "2026-09-30", "TOTAL_OPERATE_INCOME": 7777},
        ]
        result = collect_statements("sh.600519", "2026-10-01", fetch=lambda code, report: source)
        self.assertNotIn("2026-09-30", [row["报告期"] for row in result["最近四个报告期"]])
        self.assertEqual(result["最近四个报告期"][0]["报告期"], "2026-06-30")

    def test_partial_failure_preserves_income_and_does_not_infer_cash(self):
        def fetch(code, report):
            if report.endswith("CASHFLOW"):
                raise requests.ReadTimeout()
            return rows()
        result = collect_statements("600519", "2026-11-18", fetch=fetch)
        self.assertEqual(result["接口状态"]["现金流量表"]["状态"], "未取得")
        self.assertEqual(result["最近四个报告期"][0]["利润表"]["累计金额"]["营业总收入"], 300)
        self.assertEqual(result["最近四个报告期"][0]["现金流量表"], {"状态": "本报告期未取得"})

    def test_missing_previous_quarter_prevents_misleading_subtraction(self):
        source = [row for row in rows() if row["REPORT_DATE"] != "2026-06-30"]
        result = collect_statements("600519", "2026-11-18", fetch=lambda code, report: source)
        self.assertIsNone(result["最近四个报告期"][0]["利润表"]["单季度推导金额"]["营业总收入"])

    def test_fetch_is_scoped_and_has_timeout(self):
        with patch("financial_statements.requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"result": {"data": rows()}}
            fetch_rows.__wrapped__("600519", "RPT_DMSK_FN_INCOME")
            self.assertEqual(get.call_args.kwargs["params"]["filter"], '(SECURITY_CODE="600519")')
            self.assertEqual(get.call_args.kwargs["timeout"], (6, 18))
            self.assertEqual(get.call_args.kwargs["params"]["pageSize"], "16")


if __name__ == "__main__":
    unittest.main()
