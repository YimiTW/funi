#  Funi

## 運行Funi
### 前置作業
1. 下載 requirement.txt 中的所有 python 庫
2. 在 .env 文件中新增 `run_model = <model_want_to_run>`
3. 新增 .env 文件

* 若要下載大語言模型，請前往 [Hunggingface](https://huggingface.co/) ，選擇 [Models](https://huggingface.co/models) ，並選擇適合的模型
* 若要連線到 Discord ，請在 .env 文件中新增 `token = <your_token>`

### 運行模型
1. 運行 local.py 或 connect_to_discord.py ，運行大語言模型

## 個人化、微調模型
### 前置作業
1. 在 .env 文件中新增 `model_path = <model_want_to_train>` 、 `fine_tuned_model = <trained_model_output_dir>`
2. 檢視資料集中是否有不需要的訓練資料
3. 調整 fine_tune.py 中的 training_loop 到適合的次數

### 使用方法
1. 客製化 dataset.py ，透過修改變數將資料集調整成自己想要的樣子
2. 運行 fine_tune.py

## 這項專案使用的專案
* [Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) - Licensed under the LLAMA 3 COMMUNITY LICENSE

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

This project makes use of the following open-source projects:

* [Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) - Licensed under the LLAMA 3 COMMUNITY LICENSE

## 備註
* 微調llama 3基礎模型的功能尚未完成
* 自述文件尚未完成，有很多內容沒提到
* 有些資料和變數可能因為不存在於使用者的環境，從而造成錯誤
* dataset中的資料量過小，目前存在過度擬合
* online_search.py 尚未完成和實裝
* ChatGPT是很好的幫手，可以debug程式