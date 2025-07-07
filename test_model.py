import os
import sys
import warnings
import logging

# TensorFlow 경고 완전 차단 (import 전에 설정)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 사용 안함

# 표준 출력/에러 리다이렉션으로 경고 숨기기
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# 경고 메시지 완전 차단
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# TensorFlow import 시 경고 숨기기
with SuppressOutput():
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_fixed_model():
    model_path = "./outputs_gemma/final_model"
    
    print(f"Model loading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    
    # 테스트 케이스들
    test_cases = [
        "Analyze the pragma usage in the following Solidity contract snippet.",
        "Identify vulnerabilities in the following Solidity snippet using selfdestruct.",
        "What security issues exist in this smart contract?",
        "Explain the potential problems with this function."
    ]
    
    for i, instruction in enumerate(test_cases, 1):
        print(f"Question: {instruction}")
        
        # 프롬프트 생성
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # 토크나이징 (attention_mask 포함)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],  # attention_mask 명시적 전달
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1  # 반복 방지
                )
            
            # 응답 디코딩
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            print(f"Response: {response}\n")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_fixed_model()