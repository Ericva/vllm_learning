vllm serve ./QwQ-32B-AWQ \
  --host 127.0.0.1 \
  --port 12000 \
  --dtype auto \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --api-key test1