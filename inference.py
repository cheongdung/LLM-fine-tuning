# inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# checkpoint 폴더 경로 (train 시 저장된 경로 사용)
model_path = "./outputs/checkpoint-2"  # 예: outputs/checkpoint-2

# 토크나이저와 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "What is reentrancy in smart contracts?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

print("=== 질문 ===")
print(prompt)
print("=== 응답 ===")
print(tokenizer.decode(output[0], skip_special_tokens=True))
