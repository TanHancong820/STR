# STR
This repository contains the code for Semantic Temporal Rebalance-Aware Audio-Visual Event Localization.

## Our Framework
![Framework](figure/framework.png)
## Data Preparation
The AVE dataset can be downloaded from https://github.com/YapengTian/AVE-ECCV18. Other preprocessed files used in this repository can be downloaded from here. All the required data are listed as below, and these files should be placed into the folder.`data`

```text
right_label.h5    prob_label.h5      labels_noisy.h5         mil_labels.h5
train_order.h5    val_order.h5       test_order.h5
```
Note:  The extracted audio and visual features can be downloaded [here](https://pan.quark.cn/s/e8d4e3c7ae28) and they should be placed into the folder. `data`

## Fully supervised setting
* **Train:**
```bash
  python fully_supervised_main.py --model_name STR --train
```

## Weakly supervised setting
* **Train:**
```bash
  python weakly_supervised_main.py --model_name STR --train
```
Note: The pre-trained models can be downloaded [here](https://pan.quark.cn/s/a5c0bfa1d5b6) and they should be placed into the folder. `model`
