"""Read at most five official CNINFO PDFs; links alone are never treated as text."""
import io
import re
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from pypdf import PdfReader

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20
MAX_CHARS = 8000


def official_pdf_url(value):
    u = urlsplit(str(value or ""))
    if u.hostname != "static.cninfo.com.cn" or u.username or u.password or u.port not in (None, 80, 443) or not u.path.lower().endswith(".pdf"):
        raise ValueError("没有可信的巨潮PDF下载地址")
    return urlunsplit(("https", "static.cninfo.com.cn", u.path, u.query, ""))


def extract_pdf(content):
    if not content.startswith(b"%PDF"):
        raise ValueError("返回内容不是PDF")
    reader = PdfReader(io.BytesIO(content))
    total = len(reader.pages)
    # Long reports: include the beginning and ending, and disclose missing middle pages.
    pages = list(range(total)) if total <= MAX_PAGES else list(range(16)) + list(range(total-4, total))
    chunks, read_pages, failed = [], [], []
    for page in pages:
        try:
            text = re.sub(r"[ \t]+", " ", reader.pages[page].extract_text() or "").strip()
        except Exception:
            failed.append(page+1)
            continue
        if text:
            chunks.append(f"[第{page+1}页]\n{text}")
            read_pages.append(page+1)
    body = "\n\n".join(chunks)
    if not body:
        raise ValueError("PDF无可提取文字，可能是扫描件；未执行OCR")
    trimmed = len(body) > MAX_CHARS
    if trimmed:
        tail_start = len(body)-2400
        preceding = list(re.finditer(r"\[第(\d+)页\]", body[:tail_start]))
        tail_page = f"[第{preceding[-1].group(1)}页续]\n" if preceding else "[尾部摘录，页码未确定]\n"
        body = body[:5500] + "\n[中间文字未发送]\n" + tail_page + body[-2400:]
    partial = total > MAX_PAGES or trimmed or bool(failed) or len(read_pages) < total
    return {"正文状态": "部分正文" if partial else "已提取正文", "正文摘录": body,
            "PDF总页数": total, "提取到文字的页码": read_pages, "摘录包含页码": sorted({int(i) for i in re.findall(r"\[第(\d+)页", body)}), "提取失败页码": failed,
            "正文限制": "最多抽取20页并发送8000字，空白页/扫描文字/表格可能缺失；摘录不是完整审阅" if partial else "PDF文本提取可能丢失表格结构与图片，金额口径需核对"}


def fetch_document(url):
    url = official_pdf_url(url)
    started = time.monotonic()
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
        for _ in range(3):
            with session.get(url, timeout=(6, 15), stream=True, allow_redirects=False) as response:
                if response.is_redirect:
                    url = official_pdf_url(urljoin(url, response.headers.get("Location", "")))
                    continue
                response.raise_for_status()
                if int(response.headers.get("Content-Length") or 0) > MAX_BYTES:
                    raise ValueError("PDF超过10MB，保留链接供核对")
                data = bytearray()
                for chunk in response.iter_content(65536):
                    if time.monotonic()-started > 25:
                        raise TimeoutError("PDF读取超时")
                    data.extend(chunk)
                    if len(data) > MAX_BYTES:
                        raise ValueError("PDF超过10MB，停止读取")
                return {**extract_pdf(bytes(data)), "正文来源": url}
    raise ValueError("PDF重定向次数过多")


def enrich_latest(notices, fetch=fetch_document):
    selected = sorted(notices, key=lambda n: n.get("公告时间") or "", reverse=True)[:5]
    result = []
    for i, row in enumerate(selected, 1):
        item = {**row, "证据编号": f"公告{i}"}
        try:
            item.update(fetch(item.get("PDF下载链接", "")))
        except Exception as exc:
            # No exception message or headers: never leak request configuration to the AI.
            item.update(正文状态="未读取正文", 正文限制=f"下载或文本提取未成功（{type(exc).__name__}），只能依据标题与时间，链接未被AI读取")
        result.append(item)
    return result
