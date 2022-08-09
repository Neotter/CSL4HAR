'''
Date: 2022-04-13 13:27:44
LastEditors: MonakiChen
LastEditTime: 2022-08-09 21:27:20
FilePath: \CSL4HAR\parser\csl4har.py
'''
import argparse, json
from dataset.config import DatasetConfig
from dataloader.preprocessing import MaskConfig
from model.bert4cl import BERT4CLConfig
from model.gru import GRUConfig
from model.lstm import LSTMConfig
from trainer.config import PretrainConfig, TrainConfig
import os, re

def parse_args(action=None):
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

    parser.add_argument('-a', '--augmentation', choices=['spanmasking', 'clipping', 'delwords','scsense'], default='spanmasking',
                        help='choose a augmentation approach from {spanmasking, scsense, clipping, delwords}.')
    parser.add_argument( '--enable_tensor_board', default=False, action='store_true',
                        help='enable tensor board.')
    parser.add_argument('--load_path', default=False, nargs="?",
                        help='set pretrained model path')

    parser.add_argument('-m', '--model', choices=['gru', 'lstm', 'attn', 'cnn1d','cnn2d'], default='gru',
                        help='choose a downstream model from {gru, lstm, attn, cnn1d, cnn2d}.')

    args = parser.parse_args()

    # dataset config
    args.dataset_cfg = DatasetConfig.load_dataset_cfg('./config/dataset.json',args.dataset, args.dataset_version)
    # encoder and pretrain config
    args.encoder_cfg = BERT4CLConfig.from_json('./config/bert4cl.json')
    args.train_cfg = PretrainConfig.from_json('./config/pretrain.json')
    # augmentation config
    if args.augmentation == 'spanmasking':
        args.mask_cfg = MaskConfig.from_json('./config/mask.json')
        args.augment_rate = args.mask_cfg.mask_ratio
    if args.augmentation == 'delwords' or args.augmentation == 'clipping':
        args.augment_rate = 0.4
    # main action [pretrain, predict_embedding, train]
    if action == 'pretrain':
        # saved/pretrain/augmentation@augment_rate/dataset_name&version-embed_dim_lr
        save_dir = './saved/pretrain/{}@{}/{}{}-ed{}-lr{:.0e}/'.format(
            args.augmentation, args.augment_rate*100, args.dataset,args.dataset_version, args.encoder_cfg.embed_dim, args.train_cfg.lr)
    elif action == 'predict_embedding':
        # saved/embd/augmentation@augment_rate/dataset_name&version-embed_dim_lr
        save_dir = './saved/embd/{}@{}/{}{}-ed{}-lr{:.0e}/'.format(
            args.augmentation, args.augment_rate*100, args.dataset,args.dataset_version, args.encoder_cfg.embed_dim, args.train_cfg.lr)
        if args.load_path == False:
            args.load_path = './saved/pretrain/{}@{}/{}{}-ed{}-lr{:.0e}/'.format(
                args.augmentation, args.augment_rate*100, args.dataset,args.dataset_version, args.encoder_cfg.embed_dim, args.train_cfg.lr)
            # find best(maximum) epoch
            epoch_num = lambda x: int(re.findall(r'epoch(.+?)\.pth', x)[0]) if len(re.findall(r'epoch(.+?)\.pth', x))!=0 else 0
            best_epoch = max(os.listdir(args.load_path),key=epoch_num)
            args.load_path = os.path.join(args.load_path,best_epoch)
    elif action == 'train':
        # saved/classifier-augmentation@augment_rate/dataset_name&version-embed_dim_lr
        args.ds_train_cfg = TrainConfig.from_json('./config/train.json')
        if args.model == 'gru':
            args.ds_model_cfg = GRUConfig.from_json('./config/gru.json')
        if args.model =='lstm':
            args.ds_model_cfg = LSTMConfig.from_json('./config/gru.json')
        # if args.model == 'attn':
        #     args.model_cfg = BERT4CLConfig.from_json('./config/attn.json')
        # if args.model == 'cnn1d':
        #     args.model_cfg = BERT4CLConfig.from_json('./config/cnn1d.json')
        # if args.model == 'cnn2d':
        #     args.model_cfg = BERT4CLConfig.from_json('./config/cnn2d.json')
        save_dir = './saved/{}-{}@{}/{}-ed{}-lr{:.0e}/'.format(
            args.model, args.augmentation, args.augment_rate*100, args.dataset, args.encoder_cfg.embed_dim, args.train_cfg.lr)
        if args.load_path == False:
            args.load_path = './saved/embd/{}@{}/{}{}-ed{}-lr{:.0e}/'.format(
                args.augmentation, args.augment_rate*100, args.dataset,args.dataset_version, args.encoder_cfg.embed_dim, args.train_cfg.lr)
            args.load_path = os.path.join(args.load_path,'embed.npy')

    else:
        print("No action found!")
        return
    args.save_dir = save_dir
    # save config
    if args.config_path:
        args_dict = vars(args)    
        print(args.config_path)
        config = json.load(open(args.config_path,"r"))
        args_dict.update(config)
    else:
        args.config_path = './config/parser-{}.json'.format(action)
        with open(args.config_path, 'wt') as f:
            args_dict = vars(args).copy()
            args_dict.pop('config_path', None)
            args_dict.pop('dataset_cfg', None)
            args_dict.pop('encoder_cfg', None)
            args_dict.pop('mask_cfg', None)
            args_dict.pop('train_cfg', None)
            json.dump(args_dict, f, indent=4)
    return args


