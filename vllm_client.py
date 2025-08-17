from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:20800/v1",
    api_key="test1"
)
completion = client.chat.completions.create(
model="./QwQ-32B-AWQ",
messages=[
    {"role": "user", "content": "这是一个测试程序"}]
)
print(completion.choices[0].message)
