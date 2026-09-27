"""
Week 3 - 小型 parser + retry 機制（對應 project-plan-detailed.md 第3週步驟5）
延伸自 Week1 的 test_ollama.py，把單次query包裝成「呼叫 -> 嘗試parse JSON -> 失敗則retry一次」的流程，
並記錄每一筆的成功/失敗，最後可以算出格式穩定率。
"""

from __future__ import annotations

import json
import re
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

RETRY_SUFFIX = (
    "\n\n（注意：你上一次的輸出無法被解析為合法JSON。"
    "請重新輸出，這次「只能」包含一個JSON物件，不要有任何其他文字、說明或markdown標記。）"
)


def _call_ollama(prompt: str, system: str) -> dict:
    payload = {"model": MODEL, "prompt": prompt, "system": system, "stream": False}
    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    elapsed = time.time() - start
    response.raise_for_status()
    result = response.json()
    tokens_per_sec = None
    if result.get("eval_count") and result.get("eval_duration"):
        tokens_per_sec = result["eval_count"] / (result["eval_duration"] / 1e9)
    return {
        "raw_output": result["response"],
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
    }


def _extract_json(text: str) -> dict | None:
    """
    嘗試從模型輸出中取出JSON物件。
    小模型常見的偏差：在JSON前後加寒暄文字，或包在```json ... ```裡，
    所以先嘗試整段直接parse，失敗再用正規表示式抓第一個大括號區塊。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _find_invalid_enum_field(parsed: dict, enum_checks: dict[str, set[str]] | None) -> str | None:
    """回傳第一個值不在允許集合內的欄位名，全部合法則回傳None。"""
    if not enum_checks:
        return None
    for key, allowed_values in enum_checks.items():
        if key in parsed and parsed[key] not in allowed_values:
            return key
    return None


def query_with_retry(
    prompt: str,
    system: str,
    required_keys: set[str],
    enum_checks: dict[str, set[str]] | None = None,
    max_retries: int = 1,
) -> dict:
    """
    回傳統一格式：
    {
        "success": bool,
        "parsed": dict | None,
        "attempts": int,
        "elapsed_sec_total": float,
        "raw_outputs": [str, ...],   # 每次嘗試的原始輸出，方便人工檢查失敗案例
    }
    「成功」定義（Week3 pilot後修正，比原本更嚴格）：
    1. 能parse成JSON
    2. 必要欄位齊全（缺欄位視為格式不穩定）
    3. enum_checks指定的欄位，其值必須落在允許集合內
       （原本只檢查欄位存在、不檢查值是否合法，導致rating_tendency實際跑出9種用詞而非設計的3種，
       這種「有欄位但值亂跳」的狀況現在會被判定為失敗、觸發retry）。
    """
    attempts = 0
    elapsed_total = 0.0
    raw_outputs = []
    current_prompt = prompt

    while attempts <= max_retries:
        attempts += 1
        call_result = _call_ollama(current_prompt, system)
        elapsed_total += call_result["elapsed_sec"]
        raw_outputs.append(call_result["raw_output"])

        parsed = _extract_json(call_result["raw_output"])
        invalid_field = None
        if parsed is not None and required_keys.issubset(parsed.keys()):
            invalid_field = _find_invalid_enum_field(parsed, enum_checks)
            if invalid_field is None:
                return {
                    "success": True,
                    "parsed": parsed,
                    "attempts": attempts,
                    "elapsed_sec_total": round(elapsed_total, 2),
                    "raw_outputs": raw_outputs,
                }

        # 準備 retry：一般格式錯誤用通用提醒；若是「有欄位但值不合法」，
        # 額外點名是哪個欄位、允許值有哪些，比通用提醒更容易讓小模型改對
        if invalid_field is not None:
            allowed = ", ".join(sorted(enum_checks[invalid_field]))
            current_prompt = (
                prompt
                + f"\n\n（注意：你上一次輸出的\"{invalid_field}\"欄位值不合法，"
                f"此欄位只能是以下其中一個值：{allowed}，其他用詞一律不接受，請重新輸出完整JSON。）"
            )
        else:
            current_prompt = prompt + RETRY_SUFFIX

    return {
        "success": False,
        "parsed": None,
        "attempts": attempts,
        "elapsed_sec_total": round(elapsed_total, 2),
        "raw_outputs": raw_outputs,
    }
