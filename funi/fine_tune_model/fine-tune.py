import time
start_time = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import os

from dotenv import load_dotenv
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\n[ device: {device} ]\n\n")
load_dotenv()

model_path = os.getenv("model_path")
output_dir = os.getenv("fine_tuned_model")

import dataset_loader
dataset = load_dataset(dataset_loader.dataset, split="train")

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer

sft_config = SFTConfig(output_dir="/tmp")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()