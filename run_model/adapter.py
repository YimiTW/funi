import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
# load base LLM model and tokenizer

adapter_model_id = "./text-gpt-models/funi-llama3-adapter-model"

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)
def generate_response(conversation):
    input_ids = tokenizer(conversation, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    opts = model.generate(
        input_ids=input_ids, 
        max_new_tokens=256, 
        do_sample=True, 
        top_p=0.9, 
        temperature=0.9, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return opts

while True:
    inputs = input("\nYimi: ")

    conversation = f"""### Yimi:
    {inputs}

    ### Funi:
    """

    outputs = generate_response(conversation)
    print(f"\nFuni: {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(conversation):]}")