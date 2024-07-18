import json

path_list = [
    "./funi/fine_tune_model/dataset_cache.json",
    "./funi/fine_tune_model/dataset_io.json",
    "./funi/fine_tune_model/dataset_2io.json",
]
def load_json(path):
    with open(path, 'r') as f:
        dict = json.load(f)
    return dict

def get_train_split(idata, odata):
    for item in idata["train"]:
        odata.append(item)
    return odata

def get_eval_split(idata, odata):
    for item in idata["eval"]:
        odata.append(item)
    return odata

train_split = []
eval_split = []
for dataset_path in path_list:
    cache_dict = load_json(dataset_path)
    train_split = get_train_split(cache_dict, train_split)
    eval_split = get_train_split(cache_dict, eval_split)

def preprocess_chat_data(examples):
    formatted_data = []
    for sn, ss, rn, rs in zip(examples['說話者'], examples['說話者話語'], examples['回應者'], examples['回應者話語']):
        chat_str = f"{sn}: {ss}<|end_of_text|>{rn}: {rs}<|end_of_text|>"
        formatted_data.append(chat_str)
    out = tokenizer(formatted_data, max_length=512, truncation=True, padding='max_length')
    return out

dataset = {"train":train_split, "eval":eval_split}
print(dataset)