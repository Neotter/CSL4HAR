'''
Date: 2022-04-13 13:27:44
LastEditors: MonakiChen
LastEditTime: 2022-06-08 09:37:05
FilePath: \S3IMU\CODE\praser\prase_lbert_pretrain.py
'''
import argparse, json
from dataset.config import DatasetConfig
from model.gru import GRUConfig
from model.libert import LIBERTConfig
from model.lstm import LSTMConfig
from runner.config import TrainConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Scsense Main")

    ''' argument '''
    parser.add_argument('-c', '--config', default=False, nargs="?", dest="config_path",
                        help='load config from json file.')
    
    parser.add_argument('-d', '--dataset', default='hhar', choices=['mobiact', 'hhar', 'motion', 'shoaib', 'uci'],
                        help='choose a dataset from {mobiact, hhar, motion, shoaib, uci}.')
    parser.add_argument('-v','--dataset_version',  default='20_120', choices=['10_100', '20_120','20_20','20_40','20_60'],
                        help='set dataset version.')
    parser.add_argument('--seed', type=int, default=2022,
                        help='set random seed')
    parser.add_argument('-a', '--augmentation', choices=['scsense', 'spanmasking', 'clipping', 'delwords'], default='scsense',
                        help='choose a augmentation approach from {spanmasking, scsense, clipping, delwords}.')
    parser.add_argument('-m', '--model', choices=['gru', 'lstm', 'attn', 'cnn1d','cnn2d'], default='gru',
                        help='choose a downstream model from {gru, lstm, attn, cnn1d, cnn2d}.')
    parser.add_argument('-p', '--pretrained_embedding', dest="pretrained_embedding_path",
                        help='set pretrained embedding path')

    args = parser.parse_args()

    # input
    args.dataset_cfg = DatasetConfig.load_dataset_cfg('./config/dataset.json',args.dataset, args.dataset_version)
    # system
    args.pretrain_model_cfg = LIBERTConfig.from_json('./config/libert.json')
    args.train_cfg = TrainConfig.from_json('./config/pretrain.json')
    if args.model == 'gru':
        args.model_cfg = GRUConfig.from_json('./config/gru.json')
    if args.model =='lstm':
        args.model_cfg = LSTMConfig.from_json('./config/gru.json')
    if args.model == 'attn':
        args.model_cfg = LIBERTConfig.from_json('./config/attn.json')
    if args.model == 'cnn1d':
        args.model_cfg = LIBERTConfig.from_json('./config/cnn1d.json')
    if args.model == 'cnn2d':
        args.model_cfg = LIBERTConfig.from_json('./config/cnn2d.json')

    # if args.augmentation == 'spanmasking':
    #     args.mask_cfg = MaskConfig.from_json('./config/mask.json')

    # saved/downstream-augmentation/dataset_name-embed_dim_lr
    save_dir = './saved/{}-{}/{}-embed_dim{}-lr{:.0e}/'.format(
         args.model,args.augmentation, args.dataset, args.model_cfg.embed_dim, args.train_cfg.lr)
    args.save_dir = save_dir

    if args.config_path:
        args_dict = vars(args)    
        print(args.config_path)
        config = json.load(open(args.config_path,"r"))
        args_dict.update(config)
    else:
        config_path = './config/praser-train.json'
        with open(config_path, 'wt') as f:
            args_dict = vars(args).copy()
            args_dict.pop('config_path', None)
            args_dict.pop('dataset_cfg', None)
            args_dict.pop('model_cfg', None)
            args_dict.pop('pretrain_model_cfg', None)
            args_dict.pop('train_cfg', None)
            json.dump(args_dict, f, indent=4)
    return args


 