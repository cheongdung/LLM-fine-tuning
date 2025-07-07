import os
import warnings
import logging

# 경고 메시지 완전 차단
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# TensorFlow 경고 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
import json

def load_and_prepare_data():
    """데이터 로드 및 준비"""
    print("====Data loading====\n")
    
    # JSONL 파일 읽기
    data = []
    with open("data/race0_q1_q16_full_alpaca.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Data size: {len(data)}")
    return data

def create_training_data(data, tokenizer, max_length=256):
    """학습 데이터 생성"""
    print("====Tokenizing====\n")
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for item in data:
        instruction = item['instruction']
        input_text = item.get('input', '')
        output = item['output']
        
        # 프롬프트 생성
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            full_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        # 토크나이징
        full_encoded = tokenizer.encode(full_text, add_special_tokens=True)
        prompt_encoded = tokenizer.encode(prompt, add_special_tokens=True)
        
        # 길이 제한
        if len(full_encoded) > max_length:
            full_encoded = full_encoded[:max_length]
            # prompt 부분이 잘리지 않도록 확인
            if len(prompt_encoded) > max_length:
                print(f"Prompt is too long: {len(prompt_encoded)} > {max_length}")
                continue
        
        # Labels 생성 (prompt 부분은 -100)
        labels = [-100] * len(prompt_encoded) + full_encoded[len(prompt_encoded):]
        
        # 길이를 full_encoded와 맞춤
        if len(labels) > len(full_encoded):
            labels = labels[:len(full_encoded)]
        
        # 패딩 처리
        current_length = len(full_encoded)
        pad_length = max_length - current_length
        
        if pad_length > 0:
            # 패딩 추가
            input_ids = full_encoded + [tokenizer.pad_token_id] * pad_length
            attention_mask = [1] * current_length + [0] * pad_length
            labels = labels + [-100] * pad_length
        else:
            input_ids = full_encoded
            attention_mask = [1] * len(full_encoded)
        
        # 길이 확인
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length  
        assert len(labels) == max_length
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    print(f"Processed samples: {len(input_ids_list)}")
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list
    }

def main():
    print("====Training start====\n")
    
    # 모델과 토크나이저 로드
    model_name = "gpt2"
    print(f"Model loading: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    
    # 데이터 준비
    raw_data = load_and_prepare_data()
    tokenized_data = create_training_data(raw_data, tokenizer, max_length=256)
    
    # 데이터셋 생성
    dataset = Dataset.from_dict(tokenized_data)
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./outputs_gemma",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=5,
        save_steps=20,
        save_total_limit=3,
        fp16=False,
        report_to="none",
        warmup_steps=5,
        weight_decay=0.01,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # 트레이너 설정 (데이터 콜레이터 없음)
    print("====Trainer setting====\n")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # data_collator 없음 - 이미 패딩 처리했음
    )
    
    # 학습 시작
    print("====Training start====\n")
    trainer.train()
    
    # 모델 저장
    print("====Model saving====\n")
    trainer.save_model("./outputs_gemma/final_model")
    tokenizer.save_pretrained("./outputs_gemma/final_model")
    
    print("====Training complete====\n")

if __name__ == "__main__":
    main() 