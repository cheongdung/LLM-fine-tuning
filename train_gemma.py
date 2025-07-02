from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# 간단하고 안정적인 모델 선택
model_name = "gpt2"  # 가장 기본적이고 안정적인 모델

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 데이터셋 로딩
print("Loading dataset...")
dataset = load_dataset("json", data_files="data/race0_q1_q16_full_alpaca.jsonl", split="train")

def tokenize_function(examples):
    texts = [f"### Instruction:\n{inst}\n\n### Response:\n{resp}" 
             for inst, resp in zip(examples['instruction'], examples['output'])]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,  # 더 짧게 설정
        return_tensors=None
    )
    
    # labels를 input_ids와 동일하게 설정 (loss 계산을 위해)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Tokenizing...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=16,
    remove_columns=dataset.column_names
)

# 간단한 학습 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    fp16=False,  # fp16 비활성화 (에러 해결)
    report_to="none",
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

print("Setting up trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("Training...")
trainer.train()
print("Training completed!")