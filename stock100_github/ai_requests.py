"""Single-flight per browser session and audit metadata without keys or prompts."""
from copy import deepcopy
from datetime import datetime
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import threading
import time
import uuid
from zoneinfo import ZoneInfo

import requests


def estimated_cost(usage, model):
    # CNY / 1M tokens; official rates checked 2026-11-18. Not an invoice.
    rates = {"deepseek-v4-pro": (.30, 9.0, 27.0), "deepseek-flash": (.04, 2.0, 8.0)}
    if model not in rates or not all(k in usage for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens")):
        return None
    peak = sum(usage[k]*rate for k, rate in zip(("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"), rates[model]))/1e6
    return {"空闲时段估算元": round(peak/2, 6), "高峰时段估算元": round(peak, 6), "价格核对日期": "2026-11-18"}


class AnalysisRequests:
    def __init__(self, audit_path):
        self.audit_path = Path(audit_path)
        self.lock = threading.Lock()
        self.salt = secrets.token_bytes(32)
        self.entries = {}

    def signature(self, owner, url, body, key):
        stable = deepcopy(body)
        for msg in stable.get("messages", []):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                try:
                    data = json.loads(msg["content"])
                    data.pop("分析时间", None)
                    msg["content"] = data
                except (ValueError, AttributeError):
                    pass
        raw = json.dumps([owner, url, stable, key], ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hmac.new(self.salt, raw, hashlib.sha256).hexdigest()

    def audit(self, metadata):
        try:
            with self.lock:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metadata, ensure_ascii=False)+"\n")
        except OSError:
            # Failure to write a diagnostic must never resubmit a paid request.
            pass

    def run(self, owner, url, body, key, timeout, post=None):
        sig = self.signature(owner, url, body, key)
        with self.lock:
            if sig in self.entries:
                return {**deepcopy(self.entries[sig]), "reused": True}
            meta = {"local_request_id": uuid.uuid4().hex, "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "model": body.get("model"), "status": "pending"}
            self.entries[sig] = {"status": "pending", "audit": meta}
        self.audit(meta)
        started = time.monotonic()
        result = {"status": "failed", "audit": dict(meta)}
        try:
            response = (post or requests.post)(url, headers={"Authorization": "Bearer "+key}, json=body, timeout=timeout, allow_redirects=False)
            result["audit"]["http_status"] = response.status_code
            if response.status_code != 200:
                result["error"] = f"AI服务返回HTTP {response.status_code}。本次不会自动重试，请先核对账户请求记录。"
            else:
                payload = response.json()
                result["audit"]["provider_request_id"] = str(payload.get("id") or "未返回")[:160]
                raw = payload.get("usage") or {}
                usage = {k: v for k, v in raw.items() if k in {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"} and isinstance(v, int) and v >= 0}
                reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")
                if isinstance(reasoning, int) and reasoning >= 0:
                    usage["reasoning_tokens"] = reasoning
                result["audit"]["usage"] = usage
                result["audit"]["cost_estimate"] = estimated_cost(usage, body.get("model"))
                choice = payload["choices"][0]
                answer = choice["message"]["content"]
                result["audit"]["finish_reason"] = choice.get("finish_reason")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty answer")
                if choice.get("finish_reason") == "length":
                    answer += "\n\n> 服务输出达到长度上限，本报告可能未完成全部买卖条件与风险分析，请勿视为完整方案。"
                result.update(status="completed", answer=answer)
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            result.update(status="uncertain", error="连接中断、超时或返回格式异常，服务端可能已经计费；本次不会自动重试，请先核对账户请求记录。")
            result["audit"]["error_type"] = type(exc).__name__
        finally:
            result["audit"].update(status=result["status"], elapsed_seconds=round(time.monotonic()-started, 2))
            with self.lock:
                self.entries[sig] = deepcopy(result)
            self.audit(result["audit"])
        return {**result, "reused": False}
