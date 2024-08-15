import torch, os, datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
load_dotenv()

model_id = os.getenv("model_path")
output_path = "./text-gpt-models/funi-llama3-model"
check_point = "./text-gpt-models/funi-llama3-adapter-model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
use_flash_attention = False
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
	use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = "<|pad|>"
tokenizer.padding_side = "right"

#model.resize_token_embeddings(len(tokenizer))
print(f"vocab size: {len(tokenizer)}")
print(f"pad_token: {tokenizer.pad_token}")
print(f"eos_token: {tokenizer.eos_token}")

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=64, # 16 32 64 128 256
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

from transformers import TrainingArguments

args = TrainingArguments(
	output_dir=check_point,
    num_train_epochs=3, # default is 3
    per_device_train_batch_size=4, # default is 4
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5, # default is 5e-5
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True, # disable tqdm since with packing values are in correct
)

from trl import SFTTrainer

import dataset_loader
def format_instruction(sample):
	inputs = [f"""### {sn}:
{ss}

### {rn}:
{rs}{tokenizer.eos_token}""" for sn, ss, rn, rs in zip(sample["說話者"], sample["說話者話語"], sample["回應者"], sample["回應者話語"], )]
	out = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
	return out

dataset = datasets.Dataset.from_list(dataset_loader.dataset['train'])
dataset = dataset.map(format_instruction, batched=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=8192,
    args=args,
)
trainer.train()
trainer.save_model()

from peft import AutoPeftModelForCausalLM
# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    check_point,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

# testing
prompt = f"""### Yimi:
妳是誰？

### Funi:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(
	input_ids=input_ids, 
	max_new_tokens=100, 
	do_sample=True, 
	top_p=0.9, 
	temperature=0.9, 
	eos_token_id=tokenizer.eos_token_id,
	pad_token_id=tokenizer.pad_token_id,
)

print(f"adapter:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

prompt = f"""### Yimi:
誰是Yimi？

### Funi:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(
	input_ids=input_ids, 
	max_new_tokens=100, 
	do_sample=True, 
	top_p=0.9, 
	temperature=0.9, 
	eos_token_id=tokenizer.eos_token_id,
	pad_token_id=tokenizer.pad_token_id,
)

print(f"adapter:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")

prompt = f"""### Yimi:
我是誰？

### Funi:
"""

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(
	input_ids=input_ids, 
	max_new_tokens=100, 
	do_sample=True, 
	top_p=0.9, 
	temperature=0.9, 
	eos_token_id=tokenizer.eos_token_id,
	pad_token_id=tokenizer.pad_token_id,
)

print(f"adapter:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(output_path,safe_serialization=True)
tokenizer.save_pretrained(output_path)