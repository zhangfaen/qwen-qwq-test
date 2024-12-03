# pip install transformers torch pdbpp nvitop
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/QwQ-32B-Preview"

print(f"Memory allocated before model loading: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"Memory reserved before model loading: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "三角形的周长是24，三个边的长度都是整数，且任意2个边长度不相等，所有种类的三角形有多少种？"
# prompt = "Find the least odd prime factor of 2019**8 + 1 "

# 1 + 2 * 3 + 4 * 5 + 6 * (7 + 8) * 9 = 837
prompt = "在这个等式中加一对括号，使得这个等式成立: 1 + 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 = 837"

#  1 + 2 * 3 + 4 * (5 + 6 * (7 + 8)) * 9 = 3427
# prompt = "在这个等式中加几对括号，使得这个等式成立: 1 + 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 = 3427"

messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

with open('qwen_qwq_32b_vocab.txt', 'w') as f:
    for i in range(0, 151700):
        f.write(f"{i}:{tokenizer.decode([i])}\n")

# import pdb
# pdb.set_trace()

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=10000
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(response)

# Check GPU memory usage after loading the model and inference
print(f"Memory allocated after inference: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"Memory reserved after inference: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

