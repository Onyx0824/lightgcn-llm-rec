import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

def query_ollama(prompt: str, system: str = "") -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload)
    elapsed = time.time() - start
    response.raise_for_status()
    result = response.json()
    return {
        "output": result["response"],
        "elapsed_sec": round(elapsed, 2),
        "eval_count": result.get("eval_count"),
        "eval_duration_ns": result.get("eval_duration"),
    }


if __name__ == "__main__":
    system_prompt = "請務必全程使用繁體中文回答，絕對不可出現任何簡體字。"
    test_prompt = "請用一句話介紹你自己"

    result = query_ollama(test_prompt, system=system_prompt)

    print(f"輸出：{result['output']}")
    print(f"耗時：{result['elapsed_sec']} 秒")
    if result["eval_count"] and result["eval_duration_ns"]:
        tokens_per_sec = result["eval_count"] / (result["eval_duration_ns"] / 1e9)
        print(f"生成速度：{tokens_per_sec:.1f} tokens/sec")
