import time
# start pinging
start_time = time.time()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import datetime
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
model_path = os.getenv("run_model")
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
    quantization_config=bnb_config
)
# end pining
end_time = time.time()
print(f"\n[ Load take {int((end_time-start_time)*1000)}ms ]\n")

def generate_response(messages):
    input_ids = tokenizer.encode(messages, return_tensors="pt").to(device)
    
    attention_mask = (input_ids != tokenizer.eos_token_id).long().to(device)

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=1024, 
        do_sample=True, 
        temperature=0.6, 
        top_p=0.9, 
        pad_token_id=tokenizer.pad_token_id
        
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return response

def main_request(speaker_input, speaker):
    # pinging start
    start_time = time.time()
    messages_list = []
    messages_list.append({"speaker_name": speaker, "speaker_input": speaker_input})
    all_messages = "<|begin_of_text|>"
    
    for message in messages_list:
        msg = f"<|start_header_id|>{message['speaker_name']}<|end_header_id|>{message['speaker_input']}<|eot_id|>"
        all_messages += msg
    all_messages += "<|start_header_id|>Funi<|end_header_id|>"
        
    response = generate_response(all_messages)
    # pinging end
    end_time = time.time()
    print(f"\n[ping:{int((end_time - start_time)*1000)}ms]{funi_mind.funi_name}: {response}")
    # 判斷是否為discord模式
    if mode == 'discord':
        return response