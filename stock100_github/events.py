"""Conservative announcement interpretation. Evidence is data, never instructions."""
import re
import pandas as pd


def parse_event(title, body=""):
    title, body = str(title or ""), str(body or "")
    evidence = title + ("；" + body[:1500] if body else "")
    # These contexts routinely reverse a keyword-only conclusion.
    uncertain = r"终止|取消|未涉及|不涉及|不存在|否认|澄清|更正|风险提示|尚未|无法确定"
    if re.search(uncertain, evidence):
        return {"倾向": "待核实", "事件": "否定／变更／风险语境", "原因": "含否定、变更或不确定语句，不能沿用关键词方向；需核对全文及原计划。", "可信度": "低", "证据": evidence}
    if re.search(r"增持|减持|回购", title) and re.search(r"完成|完毕|届满|实施结果", title):
        return {"倾向": "待核实", "事件": "已执行／窗口结束", "原因": "披露已发生的交易或窗口结束，不等于未来新增买卖盘；需看执行比例与剩余计划。", "可信度": "低", "证据": evidence}
    rules = [
        (r"业绩预增|扭亏为盈", "业绩改善", 1, "披露利润改善，可能提高盈利预期；是否超预期仍需比较市场预期。"),
        (r"业绩预减|首亏|续亏|业绩亏损", "业绩走弱", -1, "盈利能力下降可能压低估值与买入意愿。"),
        (r"增持计划|增持股份", "股东增持", 1, "可能增加买盘并传递信心；计划不等于已执行。"),
        (r"减持计划|减持股份", "股东减持", -1, "可能增加可售股份供给；需核对数量、窗口及实际执行。"),
        (r"立案调查|行政处罚决定|债务逾期", "监管／偿债风险", -1, "可能增加经营、偿债或估值折价风险。"),
        (r"重大合同|中标", "订单", 1, "可能增加未来收入，但需核对合同金额占收入比例、利润率和履约条件。"),
    ]
    hits = [(name, direction, why) for pattern, name, direction, why in rules if re.search(pattern, title)]
    if "回购" in title:
        if re.search(r"股权激励|限制性股票|注销", evidence):
            hits.append(("回购用途待核实", 0, "可能属于激励股份回购注销，不能直接等同二级市场新增买盘。"))
        else:
            hits.append(("股份回购", 1, "可能形成买盘；需核对回购用途、金额及实际进度。"))
    signs = {direction for _, direction, _ in hits}
    direction = "偏多" if signs == {1} else "偏空" if signs == {-1} else "待核实"
    return {"倾向": direction, "事件": "、".join(x[0] for x in hits) or "未识别",
            "原因": " ".join(x[2] for x in hits) or "当前规则无法可靠判断经济影响，保留原文供查阅。",
            "可信度": "中" if body and direction != "待核实" else "低（仅标题或冲突）", "证据": evidence}


def event_report(raw):
    rows = []
    if raw is None:
        return pd.DataFrame()
    for _, row in raw.iterrows():
        title = row.get("公告标题", row.get("title", ""))
        body = row.get("正文预览", "")
        if pd.isna(body):
            body = ""
        evidence = str(title) + "；" + str(body)
        stage = "终止/变更" if re.search("终止|取消|变更", str(title)) else "实施完成" if re.search("完成|实施结果", str(title)) else "进展" if "进展" in str(title) else "计划/预告" if re.search("计划|预告|拟", str(title)) else "未确定"
        figures = re.findall(r"(?:人民币)?\s*[-+]?\d[\d,.]*(?:\s*[～至~—-]\s*\d[\d,.]*)?\s*(?:亿元|万元|万股|亿股|%)", evidence)
        rows.append({"时间": row.get("公告时间", ""), "标题": title, "进度": stage,
                     "数值片段（未核对口径）": "；".join(figures[:10]),
                     "正文状态": row.get("正文提取状态", "未取得，仅标题"),
                     **parse_event(title, body), "来源": row.get("公告详情链接", row.get("PDF下载链接", ""))})
    return pd.DataFrame(rows).drop_duplicates(subset=["标题", "时间"]) if rows else pd.DataFrame()
