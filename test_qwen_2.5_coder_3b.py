# pip install transformers torch pdbpp nvitop
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open('qwen2.5_coder_3b_vocab.txt', 'w') as f:
    for i in range(0, 151700):
        f.write(f"{i}:{tokenizer.decode([i])}\n")