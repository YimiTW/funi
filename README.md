# Funi

## 運行Funi
### 前置作業
1. 下載 requirement.txt 中的所有 python 庫
2. 將下載好的大語言模型放入 text-gpt-models 資料夾
3. 在 .env 文件中新增 `model_path = ./text-gpt-models/<path_to_model_folder>`
4. 新增 .env 文件

* 若要下載一定適用於這個專案的大語言模型，請前往 [Hunggingface](https://huggingface.co/) ，選擇 [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) ，並選擇適合的模型
* 若要連線到 Discord ，請在 .env 文件中新增 `discord_token = <your_token>`

### 運行模型
1. 運行 local.py 或 connect_to_discord.py ，運行大語言模型

## 個人化、微調模型
### 前置作業
1. 在 .env 文件中新增 `model_path = ./text-gpt-models/<path_to_model_folder>`
2. 檢視資料集中是否有不需要的訓練資料
3. 調整 fine_tune.py 中的 `num_train_epochs` `learning_rate` `per_device_train_batch_size` 到適合的值

### 使用方法
1. 客製化 dataset_io.py ，透過修改資料集將模型調整成自己想要的樣子
2. 確保下載的模型
3. 運行 fine_tune.py

* 目前fine_tune.py只有在 llama-3-8B (基礎模型) 上證實有效

## License

This project (My code, not llama-3-8B) is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

This project makes use of the following open-source projects:

* [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) - Licensed under the LLAMA 3 COMMUNITY LICENSE

## 備註
* 微調llama 3基礎模型的功能尚未完成
* 自述文件尚未完成，有很多內容沒提到
* 有些資料和變數可能因為不存在於使用者的環境，從而造成錯誤
* dataset中的資料量過小，目前存在過度擬合
* online_search.py 尚未完成和實裝
* ChatGPT是很好的幫手，可以debug程式