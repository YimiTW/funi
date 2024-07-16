# fine-tune will not update for a long time
import time
start_time = time.time()

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig,DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os
from dotenv import load_dotenv
# check if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\n[ device: {device} ]\n\n")

# use your model path
load_dotenv()
model_path = os.getenv("model_path")
output_dir = os.getenv("fine_tuned_model")

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
training_args = TrainingArguments(
    num_train_epochs=100,           # 训练的轮数
    output_dir="./results",         # 保存模型和其他输出的目录
    per_device_train_batch_size=1,  # 每个设备的训练批次大小
    per_device_eval_batch_size=2,   # 每个设备的评估批次大小
    logging_dir="./logs",           # 日志文件保存目录
    logging_steps=10,               # 每隔多少步记录一次日志
    fp16=True,                       # 启用FP16训练
    gradient_accumulation_steps=16
) # 訓練參數設置

# Tokenizer 加載
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
) # 加載4-bit量化模型

model = get_peft_model(model, lora_config) # 應用QLoRA到模型

# load dataset
import dataset_loader
train_dataset = Dataset.from_list(dataset_loader.dataset['train'])
eval_dataset = Dataset.from_list(dataset_loader.dataset['eval'])

def apply_multi_role_chat_template(examples):
    formatted_data = []
    for sn, ss, rn, rs in zip(examples['說話者'], examples['說話者話語'], examples['回應者'], examples['回應者話語']):
        chat_str = f"<|begin_of_text|><|start_header_id|>{sn}<|end_header_id|>{ss}<|eot_id|><|start_header_id|>{rn}<|end_header_id|>{rs}<|end_of_text|>"
        formatted_data.append(chat_str)
    print(formatted_data)
    return formatted_data

def preprocess_chat_data(examples):
    formatted_data = apply_multi_role_chat_template(examples)
    out = tokenizer(formatted_data, max_length=512, truncation=True, padding='max_length')
    return out

tokenized_train_datasets = train_dataset.map(preprocess_chat_data, batched=True) # Preprocess the train dataset
tokenized_eval_datasets = eval_dataset.map(preprocess_chat_data, batched=True) # Preprocess the eval dataset
# end
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
) # 定義Trainer

end_time = time.time()
print(f"\n\n[ load take {int((end_time-start_time)*1000)}ms ]\n\n")
start_time = time.time()
trainer.train() # 微調模型
# 合併模型
model = model.merge_and_unload()
# 保存微調後的模型
model.save_pretrained(output_dir)
# Tokenizer 也要跟著另外存一份
tokenizer.save_pretrained(output_dir)
# 評估模型（可選）
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

end_time = time.time()
print(f"\n\n[ train take {int((end_time-start_time))}s ]\n\n")