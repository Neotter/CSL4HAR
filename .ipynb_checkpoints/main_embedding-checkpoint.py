import os
import random
from time import time
import numpy as np
import pandas as pd
from model.bert4cl import BERT4CL

from dataloader.embedding import *
from praser.embedding import parse_args
from utils.log_helper import *
from utils.get_device import *
from utils.model_helper import *
from utils.plot_helper import plot_embedding
import warnings
warnings.filterwarnings("ignore")



def get_embedding(args):
    # log_path
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    dataloader = DatasetSelector4Embedding(args)

    data_loader = dataloader.get_dataloader()
    
    # construct model & optimizer
    input_dim = dataloader.dataset_cfg.dimension
    embed_dim = args.model_cfg.embed_dim

    decoder = None
    model = BERT4CL(args.model_cfg, input_dim, embed_dim, following_model=decoder)
    optimizer = None

    logging.info(model)
    
    model = load_model(model, args.pretrained_model_path, load_self=False)
    
    logging.info(model)
    model.to(device)
    
    time0 = time()
    embeds = []
    labels = []

    model.eval()

    for iter, batch in enumerate(data_loader):
        batch = map(lambda x: x.to(device), batch)
        seqs, label = batch
        with torch.no_grad():
            embed = model(seqs, mode='pretrain_predict')
            embeds.append(embed)
            labels.append(label)

    embeds = torch.cat(embeds, 0).cpu().numpy()
    np.save(os.path.join(args.save_dir,'embed.npy'), embeds)
    return dataloader, embeds

def load_embedding_label(model_file, dataset, dataset_version):
    embed_name = 'embed_' + model_file + '_' + dataset + '_' + dataset_version
    label_name = 'label_' + dataset_version
    embed = np.load(os.path.join('embed', embed_name + '.npy')).astype(np.float32)
    labels = np.load(os.path.join('dataset', dataset, label_name + '.npy')).astype(np.float32)
    return embed, labels

if __name__ == "__main__":
    args = parse_args()
    dataloader, embeds = get_embedding(args)

    label_index = 0
    label_names, label_num = dataloader.label_names, dataloader.label_num
    data_tsne, labels_tsne = plot_embedding(embeds, dataloader.label, label_index=label_index, reduce=1000, label_names=label_names)

