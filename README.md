# STR
This repository contains the code for Semantic Temporal Rebalance-Aware Audio-Visual Event Localization.

## Our Framework
![Framework](figures/framework.png)
## Data Preparation
The AVE dataset can be downloaded from https://github.com/YapengTian/AVE-ECCV18. Other preprocessed files used in this repository can be downloaded from here. All the required data are listed as below, and these files should be placed into the folder.`data`

```text
right_label.h5    prob_label.h5      labels_noisy.h5         mil_labels.h5
train_order.h5    val_order.h5       test_order.h5
