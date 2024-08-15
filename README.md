# Funi

## 運行Funi
### 前置作業
1. 下載 requirement.txt 中的所有 python 庫
2. 將下載好的大語言模型放入 text-gpt-models 資料夾

* 若要下載一定適用於這個專案的大語言模型，請前往 [Hunggingface](https://huggingface.co/) ，選擇 [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* 若要連線到 Discord ，請在 .env 文件中新增 `discord_token = <your_token>`

### 微調模型
1. 新增 .env 文件
2. 在 .env 文件中新增 `model_path = ./text-gpt-models/<path_to_model_folder>`
3. 檢視 dataset_io.py 中是否有不需要、需要新增或修改的訓練資料
4. 調整 fine_tune.py 中的 `num_train_epochs` `learning_rate` `per_device_train_batch_size` 到適合的值
5. 運行 fine_tune.py

* 如果想讓model有其他名子，可以修改 fine_tune.py 中的 `{rn}` 成名子，記得下方測試adapter model 的 `prompt` 和 funi.py 中的 `conversation` 的 名子都要改掉
* 目前fine_tune.py只有在 Llama-3-8B (基礎模型) 上證實有效

### 運行模型
1. 運行 local.py 或 connect_to_discord.py ，運行大語言模型

* Llama-3-8B 模型需要至少 7GB 的 vram 才能穩定運行

## License

This project (My code, not llama-3-8B) is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

This project makes use of the following open-source projects:

* [GPT-Sovits](https://github.com/RVC-Boss/GPT-SoVITS) - Licensed under the MIT LICENSE
* Built with Meta [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) - Licensed under the LLAMA 3 COMMUNITY LICENSE

## 備註
* 自述文件可能會有遺漏的地方，如果照做不會動記得 bug report
* online_search.py 尚未完成和實裝
* ChatGPT是很好的幫手，可以debug程式