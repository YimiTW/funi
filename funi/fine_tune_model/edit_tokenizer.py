from transformers import PreTrainedTokenizerFast
import os
from dotenv import load_dotenv
load_dotenv()
model = os.getenv("model_path")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
SPECIAL_TOKENS = {
    'bos_token': '<|begin_of_text|>',
    'eos_token': '<|end_of_text|>',
}
tokenizer.add_special_tokens(SPECIAL_TOKENS)