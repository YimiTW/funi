import time
start_time = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import os

from dotenv import load_dotenv
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\n[ device: {device} ]\n\n")
load_dotenv()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
) # quantization config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任務類型
    inference_mode=False,          # 訓練模式
    r=8,                           # Low-rank adaptation 參數
    lora_alpha=32,                 # LoRA 放大係數
    lora_dropout=0.1,              # LoRA dropout率
    bias="none",
) # 定義QLoRA配置

model_path = os.getenv("model_path")
output_dir = os.getenv("fine_tuned_model")

import dataset_loader
dataset = load_dataset(dataset_loader.dataset['train'])

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)
model = get_peft_model(model, lora_config)

sft_config = SFTConfig(output_dir="/tmp")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()

model = model.merge_and_unload()
# 保存微調後的模型
model.save_pretrained(output_dir)
# Tokenizer 也要跟著另外存一份
tokenizer.save_pretrained(output_dir)