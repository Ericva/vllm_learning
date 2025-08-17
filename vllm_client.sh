curl http://localhost:20800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test1" \
  -d '{
    "model": "./QwQ-32B-AWQ",
    "messages": [
      {"role": "user", "content": "这是一个测试程序"}
    ]
  }'