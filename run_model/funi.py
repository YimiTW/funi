import time
# start pinging
start_time = time.time()

from gpt_sovits.gpt_sovits import inference_webui
from pydub.playback import play

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[ device: {device} ]\n")

# Initialize model and tokenizer
mode = 'local'
model_path = "./text-gpt-models/funi-llama3-model"
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

def generate_response(conversation):
    input_ids = tokenizer(conversation, truncation=True, return_tensors="pt").input_ids.cuda()
    
    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=256, 
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(conversation):]
    #print(outputs[0][input_ids.shape[-1]:]) # check tensor
    return response

def main_request(speaker_input, speaker):
    # pinging start
    start_time = time.time()
    conversation = f"""### {speaker}:
{speaker_input}

### Funi:
"""
    response = generate_response(conversation)
    response = response.split("\n\n###")
    response = response[0]

    # generate sound
    audio_segment = inference_webui.tts_request(
        inp_ref="/home/yimi/Downloads/test sovit 1-2.m4a",
        prompt_text="",
        prompt_language="中文",
        text=response, # input response
        text_language="多语种混合",
        how_to_cut="凑四句一切",
        top_k=5,
        top_p=1.0,
        temperature=1.0,
        ref_text_free=True # 无参考文本模式
    )
    # pinging end
    end_time = time.time()

    print(f"\n[ping:{int((end_time - start_time)*1000)}ms]Funi: {response}")
    # 播放音频
    play(audio_segment)



    # 判斷是否為discord模式
    if mode == 'discord':
        return response