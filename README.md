<!--
 * @Date: 2022-08-05 20:13:28
 * @LastEditors: MonakiChen
 * @LastEditTime: 2022-08-09 23:21:42
 * @FilePath: \CSL4HAR\README.md
-->
# CSL4HAR
A contrastive representation learning framework for human activity recognition.
This frameworks derive the self-supervised signal from the contrastive objective to improve the preformance of the representation learning.
With the representations learned under easy-to-access unlabeled data, downstream models finetuned with limited labeled samples can perform promising results. 

---

# Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
- pytorch == 1.12.0
- python == 3.8.0
- scikit-learn == 1.1.1
- scipy == 1.8.1
- seaborn == 0.11.2

# Usage
To demonstrate the reproducibility of the best performance reported in our paper and faciliate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (might be different for the custormized datasets) in the scripts.

The instruction of commands has been clearly stated in the codes (see the parse_args function in parse/csl4har.py).

## Pertrain encoder

Example:
Pretrain encoder with UCI dataset.
The spanmasking is used as data augmentation method.
```bash
python main_pretrain.py -d uci -v 20_120 -a spanmasking
```


## Output embeddings

Example: Output all UCI samples' embeddings by pretrained encoder.
```
python main_embedding.py -d uci -v 20_120 -a spanmasking
```


## Finetune downstream model 
Example: Finetune downstream model with a small number of labeled data.
```
python main_train.py -d uci -v 20_120 -a spanmasking
```

## Common parameter:
```
usage: main_pretrain.py [-h] [-c [CONFIG_PATH]] [-d {mobiact,hhar,motion,shoaib,uci}] [-v {10_100,20_120,20_20,20_40,20_60}] [--seed SEED] [-a {spanmasking,clipping,delwords,scsense}] [--enable_tensor_board]
                        [--load_path [LOAD_PATH]] [-m {gru,lstm,attn,cnn1d,cnn2d}]

CSL4HAR Main

optional arguments:
  -h, --help            show this help message and exit
  -c [CONFIG_PATH], --config [CONFIG_PATH]
                        load config from json file.
  -d {mobiact,hhar,motion,shoaib,uci}, --dataset {mobiact,hhar,motion,shoaib,uci}
                        choose a dataset from {mobiact, hhar, motion, shoaib, uci}.
  -v {10_100,20_120,20_20,20_40,20_60}, --dataset_version {10_100,20_120,20_20,20_40,20_60}
                        set dataset version.
  --seed SEED           set random seed
  -a {spanmasking,clipping,delwords,scsense}, --augmentation {spanmasking,clipping,delwords,scsense}
                        choose a augmentation approach from {spanmasking, scsense, clipping, delwords}.
  --enable_tensor_board
                        enable tensor board.
  --load_path [LOAD_PATH]
                        set pretrained model path
  -m {gru,lstm,attn,cnn1d,cnn2d}, --model {gru,lstm,attn,cnn1d,cnn2d}
                        choose a downstream model from {gru, lstm, attn, cnn1d, cnn2d}.
```

# Dataset
We provide three processed datasets: UCI, MotionSense, HHAR. 
- You can find the full version of datasets via [UCI](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions), [MotionSense](https://github.com/mmalekzadeh/motion-sense), [HHAR](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition)
- The MotionAct dataset are available upon request (email Biomedical Informatics Laboratory at: bmi@hmu.gr), after the sign of a database usage agreement which establish the terms and conditions of data usage.
- We follow [LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public) to preprocess datasets.

| Dataset      | MobiAct          | UCI       | HHAR      | Motion    |
| ------------ | ---------------- | --------- | --------- | --------- |
| # Users      | 66               | 30        | 9         | 24        |
| # Activities | 16               | 6         | 6         | 6         |
| # Samples    | 3200             | 2088      | 9166      | 4535      |
| Sensor Type  | Acc, Gyro, Orien | Acc, Gyro | Acc, Gyro | Acc, Gyro |

# Configuration
All configuration is in config/.

- pretrain.json
  - Settings of pretraining stage.
- train.json
  - Settings of downstream training stage.
- parser-[pretrain,predict_embedding,train].json
  - The stored configuration of each parser.
- bert4cl.json
  - Settings of bert encoder.
- mask.json
  - Settings of span mask augmentation.
- dataset.json
  - Infomation of each processed dataset.
- gru.json | lstm.json | cnn1d.json | cnn2d.json
  - Settings of downstream models.