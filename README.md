# CSL4HAR
A contrastive representation learning framework for human activity recognition.
This frameworks derive the self-supervised signal from the contrastive objective to improve the preformance of the representation learning.
With the representations learned under easy-to-access unlabeled data, downstream models finetuned with limited labeled samples can perform promising results. 

---
# Usage
## Pertrain encoder

Example:
Pretrain encoder with UCI dataset.
The spanmasking is used as data augmentation method.
```bash
python main_pretrain.py -d uci -v 20_120 -a spanmasking
```
Optional arguments:
```bash
  -h, --help            show this help message and exit
  -c [CONFIG_PATH], --config [CONFIG_PATH]
                        load config from json file.
  -d {mobiact,hhar,motion,shoaib,uci}, --dataset {mobiact,hhar,motion,shoaib,uci}
                        choose a dataset from {mobiact, hhar, motion, shoaib, uci}.
  -v {10_100,20_120,20_20,20_40,20_60}, --dataset_version {10_100,20_120,20_20,20_40,20_60}
                        set dataset version.
  --seed SEED           set random seed
  -a {spanmasking,clipping,delwords}, --augmentation {spanmasking,clipping,delwords}
                        choose a augmentation approach from {spanmasking, scsense, clipping, delwords}.
```

## Output embeddings

Example: Output all UCI samples' embeddings by pretrained encoder.
```
python main_embedding.py -d uci -v 20_120 -a spanmasking -p /saved/pretrain/uci20_120-embed_dim72-lr1e-04/model_epoch3050.pth
```

Optional arguments:
```bash
  -h, --help            show this help message and exit
  -c [CONFIG_PATH], --config [CONFIG_PATH]
                        load config from json file.
  -d {mobiact,hhar,motion,shoaib,uci}, --dataset {mobiact,hhar,motion,shoaib,uci}
                        choose a dataset from {mobiact, hhar, motion, shoaib, uci}.
  -v {10_100,20_120,20_20,20_40,20_60}, --dataset_version {10_100,20_120,20_20,20_40,20_60}
                        set dataset version.
  --seed SEED           set random seed
  -a {scsense,spanmasking,clipping,delwords}, --augmentation {scsense,spanmasking,clipping,delwords}
                        choose a augmentation approach from {spanmasking, scsense, clipping, delwords}.
  -p PRETRAINED_MODEL_PATH, --pretrained_model PRETRAINED_MODEL_PATH
                        set pretrained model path
```

## Finetune downstream model 
Example: Finetune downstream model with a small number of labeled data.
```
python main_train.py -d uci -v 20_120 -a spanmasking -p ./saved/embedding/spanmasking@40.0/uci-embed_dim72-lr1e-04/embed.npy
```
optional arguments:
```
  -h, --help            show this help message and exit
  -c [CONFIG_PATH], --config [CONFIG_PATH]
                        load config from json file.
  -d {mobiact,hhar,motion,shoaib,uci}, --dataset {mobiact,hhar,motion,shoaib,uci}
                        choose a dataset from {mobiact, hhar, motion, shoaib, uci}.
  -v {10_100,20_120,20_20,20_40,20_60}, --dataset_version {10_100,20_120,20_20,20_40,20_60}
                        set dataset version.
  --seed SEED           set random seed
  -a {scsense,spanmasking,clipping,delwords}, --augmentation {scsense,spanmasking,clipping,delwords}
                        choose a augmentation approach from {spanmasking, scsense, clipping, delwords}.
  -m {gru,lstm,attn,cnn1d,cnn2d}, --model {gru,lstm,attn,cnn1d,cnn2d}
                        choose a downstream model from {gru, lstm, attn, cnn1d, cnn2d}.
  -p PRETRAINED_EMBEDDING_PATH, --pretrained_embedding PRETRAINED_EMBEDDING_PATH
                        set pretrained embedding path
```

