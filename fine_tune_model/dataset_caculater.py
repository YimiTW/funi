import dataset_loader
dataset_list = dataset_loader.dataset

print(f"train_len: {len(dataset_list['train'])}, eval_len: {len(dataset_list['eval'])}")