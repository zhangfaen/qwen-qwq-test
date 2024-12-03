# pip install transformers torch pdbpp nvitop qwen_vl_utils
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



print(f"Memory allocated before model loading: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"Memory reserved before model loading: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


tokenizer = processor.tokenizer


with open('qwen2_vl_7b_vocab.txt', 'w') as f:
    for i in range(0, 151700):
        f.write(f"{i}:{tokenizer.decode([i])}\n")


# Check GPU memory usage after loading the model and inference
print(f"Memory allocated after inference: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"Memory reserved after inference: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

