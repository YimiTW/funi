import json
dataset_path = "./funi/fine_tune_model/dataset_2.json"
def load_json(path):
    with open(path, 'r') as f:
        text = json.load(f)
    return text
dataset = load_json(dataset_path)