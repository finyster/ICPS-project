import requests
import json

def ask_llama(prompt, model="llama3:8b"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    print(f"==> 送出 prompt: {prompt}")
    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True)
        resp.raise_for_status()
        print("=== LLM 回應 ===")
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                print(data.get("response", ""), end="", flush=True)
        print()  # 換行
    except Exception as e:
        print("❌ 發生錯誤:", e)

if __name__ == "__main__":
    ask_llama("用三句話介紹 LLaMA 3")
