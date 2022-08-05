'''
Date: 2022-04-13 13:27:44
LastEditors: MonakiChen
LastEditTime: 2022-08-05 21:56:28
FilePath: \CSL4HAR\praser\pretrain.py
'''
import argparse, json
from dataset.config import DatasetConfig
from dataloader.preprocessing import MaskConfig
from model.libert import LIBERTConfig
from runner.config import SCSensePretrainConfig

def parse_args():
    parser = argparse.ArgumentParser(description="CSL4HAR Main")

    ''' argument '''
    parser.add_argument('-c', '--config', default=False, nargs="?", dest="config_path",
                        help='load config from json file.')
    
    parser.add_argument('-d', '--dataset', default='hhar', choices=['mobiact', 'hhar', 'motion', 'shoaib', 'uci'],
                        help='choose a dataset from {mobiact, hhar, motion, shoaib, uci}.')
    parser.add_argument('-v','--dataset_version',  default='20_120', choices=['10_100', '20_120','20_20','20_40','20_60'],
                        help='set dataset version.')
    parser.add_argument('--seed', type=int, default=2022,
                        help='set random seed')
    parser.add_argument('-a', '--augmentation', choices=['spanmasking', 'clipping', 'delwords'], default='spanmasking',
                        help='choose a augmentation approach from {spanmasking, scsense, clipping, delwords}.')

    args = parser.parse_args()

    # input
    args.dataset_cfg = DatasetConfig.load_dataset_cfg('./config/dataset.json',args.dataset, args.dataset_version)
    # system
    args.model_cfg = LIBERTConfig.from_json('./config/libert.json')
    args.train_cfg = SCSensePretrainConfig.from_json('./config/pretrain.json')

    if args.augmentation == 'spanmasking':
        args.mask_cfg = MaskConfig.from_json('./config/mask.json')
        args.augment_rate = args.mask_cfg.mask_ratio
    if args.augmentation == 'delwords' or args.augmentation == 'clipping':
        args.augment_rate = 0.5

    # trained_model/pretrain/augmentation/dataset_name-embed_dim_lr
    save_dir = './saved/pretrain/{}@{}/{}{}-embed_dim{}-lr{:.0e}/'.format(
         args.augmentation, args.augment_rate*100, args.dataset,args.dataset_version, args.model_cfg.embed_dim, args.train_cfg.lr)
    args.save_dir = save_dir

    if args.config_path:
        args_dict = vars(args)    
        print(args.config_path)
        config = json.load(open(args.config_path,"r"))
        args_dict.update(config)
    else:
        config_path = './config/praser-pretrain.json'
        with open(config_path, 'wt') as f:
            args_dict = vars(args).copy()
            args_dict.pop('config_path', None)
            args_dict.pop('dataset_cfg', None)
            args_dict.pop('model_cfg', None)
            args_dict.pop('train_cfg', None)
            json.dump(args_dict, f, indent=4)
    return args


