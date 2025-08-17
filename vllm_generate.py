from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# 1. 要推理的 prompt
raw_prompts = [
    "北京奥运会中国得了多少块金牌?第一块金牌和最后一块金牌是谁获得的",
    "李白写过最好的诗句是什么?他和孟浩然是什么关系",
    "李白是什么民族的人"
]

# 2.加载prompt
model_name = "./QwQ-32B-AWQ"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
messages_list =[[{"role":"user","content":prompt}] for prompt in raw_prompts]
#print("messages_list: ", messages_list)
prompts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    for messages in messages_list
]
print("prompts: ", prompts, type(prompts))

# 3. 采样参数
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=1024
)

# 4. 初始化 vLLM 引擎
llm = LLM(
    model="./QwQ-32B-AWQ",           # 本地权重路径
    tensor_parallel_size=1,          # 2 张 GPU 并行
    max_model_len=32768,             # 最大上下文长度
    enable_prefix_caching=True       # 开启前缀 KV-Cache 复用
)

# 5. 推理
outputs = llm.generate(prompts, sampling_params)

# 6. 简单后处理
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print("Prompt:", prompt)
    print("Answer:", generated.strip())
    print("-" * 40)