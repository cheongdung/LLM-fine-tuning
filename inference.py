from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Fine-tuned 모델이 저장된 경로
model_path = "./outputs"  # train_gemma.py에서 저장한 디렉토리와 동일하게

# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# 프롬프트 입력 (여기 수정해서 질문 바꿔도 됨)
prompt = "What is reentrancy in smart contracts?"

# 토크나이징
inputs = tokenizer(prompt, return_tensors="pt")

# GPU 사용 가능하면 GPU로 보내기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 텍스트 생성
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

# 출력 디코딩
decoded = tokenizer.decode(output[0], skip_special_tokens=True)

# 출력 결과 보기
print("=== 질문 ===")
print(prompt)
print("=== 응답 ===")
print(decoded)
