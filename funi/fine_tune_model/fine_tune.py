import torch, os, datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
load_dotenv()

model_id = os.getenv("model_path")

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
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
 
# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
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
    output_dir="llama-7-int4-dolly",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    #learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True # disable tqdm since with packing values are in correct
)

from trl import SFTTrainer
 
max_seq_length = 2048 # max sequence length for model and packing of the dataset

import dataset_loader
def format_instruction(sample):
	inputs = [f"""### {sn}:
{ss}
 
### {rn}:
{rs}
""" for sn, ss, rn, rs in zip(sample["說話者"], sample["說話者話語"], sample["回應者"], sample["回應者話語"], )]
	print(inputs)
	out = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
	return out

dataset = datasets.Dataset.from_list(dataset_loader.dataset['train'])
dataset = dataset.map(format_instruction, batched=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=False,
    args=args,
)

trainer.train()
trainer.save_model()

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
 
args.output_dir = "llama-7-int4-dolly"
 
# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

from datasets import load_dataset
 
prompt = f"""### Yimi:
Hi
 
### Funi:
"""
 
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)
 

print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")