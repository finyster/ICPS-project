# test_openai_client_llama.py
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 內容隨意
)

def test_basic_chat():
    resp = client.chat.completions.create(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": "你是一位中文助理"},
            {"role": "user", "content": "告訴我三個台灣特色美食"}
        ],
        max_tokens=128,
        stream=False
    )
    print("Assistant:", resp.choices[0].message.content)

if __name__ == "__main__":
    test_basic_chat()
