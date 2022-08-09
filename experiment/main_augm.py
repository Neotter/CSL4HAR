'''
Date: 2022-08-09 22:24:50
LastEditors: MonakiChen
LastEditTime: 2022-08-09 23:14:48
FilePath: \CSL4HAR\experiment\main_augm.py
'''
# from ..main_pretrain import pretrain
from .parse.csl4har import parse_args


if __name__ == "__main__":
    args = parse_args('pretrain')
    # args.mask_cfg = argscd .mask_cfg._replace(mask_ratio=0.5)
    print(args)