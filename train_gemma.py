from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch

model_name = "google/gemma-2b"

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

#데이터셋 로딩
dataset = load_dataset("json", data_files="race0_q1_q16_full_alpaca.jsonl", split="train")

def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=64)
    tokens["labels"]=tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

#학습 설정
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    fp16=False,
    bf16=True,
    report_to="none")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
)

trainer.train()