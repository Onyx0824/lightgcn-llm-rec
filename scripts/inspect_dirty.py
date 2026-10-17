"""檢查 dirty items 是被哪些字元判定的（Week 5 去噪前的抽查）。
用法（專案根目錄）：python scripts/inspect_dirty.py [--file results/week4/items.jsonl] [--show 15]
"""
import argparse
import json
import re
from collections import Counter

# 與 src/llm_augmentation/run_week4_generate.py 的 has_bad_text 同一套判定，但改成回傳「哪些字」
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "llm_augmentation"))
from run_week4_generate import SIMPLIFIED_ONLY  # noqa: E402


def bad_chars(parsed):
    text = json.dumps(parsed, ensure_ascii=False)
    out = []
    for ch in text:
        o = ord(ch)
        if ch == "\ufffd" or ch in SIMPLIFIED_ONLY or 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF:
            out.append(ch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/week4/items.jsonl")
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
    dirty = [r for r in rows if r.get("dirty")]
    retry = [r for r in rows if r.get("attempts", 1) > 1]
    dirty_ids = {id(r) for r in dirty}
    print(f"總筆數 {len(rows)} | dirty {len(dirty)} | attempts>1 {len(retry)} | dirty ⊂ attempts>1: "
          f"{all(r.get('attempts', 1) > 1 for r in dirty)}")

    cnt, per_item = Counter(), []
    for r in dirty:
        if r.get("parsed") is None:
            continue
        chars = bad_chars(r["parsed"])
        cnt.update(set(chars))              # 每個 item 每個字只算一次
        per_item.append((r["item_idx"], chars))
    print("\n觸發 dirty 的字元（出現在幾個 item）：")
    for ch, n in cnt.most_common(15):
        print(f"  {ch!r} U+{ord(ch):04X}: {n}")
    print(f"\n只靠單一字元就被判 dirty 的 item：{sum(1 for _, c in per_item if len(set(c)) == 1)}/{len(per_item)}")
    print(f"\n前 {args.show} 筆（item_idx → 觸發字元）：")
    for idx, chars in per_item[:args.show]:
        print(f"  {idx}: {''.join(sorted(set(chars)))}")


if __name__ == "__main__":
    main()
