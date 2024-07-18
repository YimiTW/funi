import time
# start pinging
start_time = time.time()

from transformers import AutoTokenizer, AutoModelForCausalLM#, BitsAndBytesConfig
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
model_path = os.getenv("model_path")
chat_data_all_path = "./chat_data/chat_data_all.json"
chat_data_all_backup_path = "./chat_data/chat_data_all_backup.json"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# end pining
end_time = time.time()
print(f"\n[ Load take {int((end_time-start_time)*1000)}ms ]\n")

def generate_response(messages):
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
        ).to("cuda")
    
    outputs = model.generate(
        input_ids, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.6, 
        top_p=0.9, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return response

def main_request(speaker_input, speaker):
    # pinging start
    start_time = time.time()
    messages_list = []
    messages_list.append({"role": speaker, "content": speaker_input})
        
    response = generate_response(messages_list)
    # pinging end
    end_time = time.time()
    print(f"\n[ping:{int((end_time - start_time)*1000)}ms]{funi_mind.funi_name}: {response}")
    # 判斷是否為discord模式
    if mode == 'discord':
        return response