import time
# start pinging
start_time = time.time()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch

# funi's mind
import funi_mind

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[ device: {device} ]\n")

from dotenv import load_dotenv
load_dotenv()

# Initialize model and tokenizer
mode = 'local'
model_path = os.getenv("fine_tuned_model")
chat_data_all_path = "./chat_data/chat_data_all.json"
chat_data_all_backup_path = "./chat_data/chat_data_all_backup.json"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)
# end pining
end_time = time.time()
print(f"\n[ Load take {int((end_time-start_time)*1000)}ms ]\n")
print(tokenizer.pad_token)
def generate_response(conversation):
    input_ids = tokenizer(conversation, truncation=True, return_tensors="pt").input_ids.cuda()
    
    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=100, 
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

def main_request(speaker_input, speaker):
    # pinging start
    start_time = time.time()
    conversation = f"""### {speaker}:
{speaker_input}

### Funi:
"""
    response = generate_response(conversation)
    # pinging end
    end_time = time.time()
    print(f"\n[ping:{int((end_time - start_time)*1000)}ms]{funi_mind.funi_name}: {response}")
    # 判斷是否為discord模式
    if mode == 'discord':
        return response