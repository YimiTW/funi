import json

path_list = [
    "./fine_tune_model/dataset_io.json",
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
    eval_split = get_eval_split(cache_dict, eval_split)

dataset = {"train":train_split, "eval":eval_split}