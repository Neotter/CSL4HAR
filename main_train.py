import sys
import random
from itertools import chain
from time import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from dataloader.pretrain import DatasetSelector4Pretrain
from dataloader.train import DatasetSelector4Train
from model.attn import ATTN

from model.bert4cl import *
from model.cnn1d import CNN1D
from model.cnn2d import CNN2D
from model.decoder_projhead import ProjectHead
from model.gru import GRU
from model.lstm import LSTM
from praser.train import *
from utils.log_helper import *
from utils.get_device import *
from utils.metrics import calc_acc, calc_confus_matric, calc_f1
from utils.model_helper import *
from torch.utils.tensorboard import SummaryWriter 

def train(args):
    # log_path
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # tensorboard
    writer = SummaryWriter(args.save_dir)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    dataloader = DatasetSelector4Train(args)
    data_loader_train, data_loader_vali, data_loader_test = dataloader.get_dataloader()

    logging.info('Balance Label Size: %d; Unlabel Size: %d; Real Label Rate: %0.3f;' % (data_loader_train.dataset.labels.shape[0], data_loader_vali.dataset.labels.shape[0], data_loader_test.dataset.labels.shape[0] * 1.0 / dataloader.labels.shape[0]))
    
    logging.info('Train Size: {}, Vali Size: {}'.format(data_loader_train.dataset.__len__(), data_loader_vali.dataset.__len__()))


    # construct model & optimizer
    if hasattr(args, 'pretrained_embedding_path'):
        input_dim = 72
    else:
        input_dim = dataloader.dataset_cfg.dimension
    output_dim = args.dataset_cfg.dimension

    if args.model == 'gru':
        model = GRU(args.model_cfg, input_dim, output_dim)
    if args.model == 'lstm':
        model = LSTM(args.model_cfg, input_dim, output_dim)
    if args.model == 'cnn1d':
        model = CNN1D(args.model_cfg, input_dim, output_dim)
    if args.model == 'cnn2d':
        model = CNN2D(args.model_cfg, input_dim, output_dim)
    if args.model == 'attn':
        model = ATTN(args.model_cfg, input_dim, output_dim)
        
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.train_cfg.lr)

    logging.info(model)
    logging.info(optimizer)

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    metrics_list = {'acc': [],'f1': [], 'confus_matrix': []}

    time0 = time()
    for epoch in range(1, args.train_cfg.n_epochs+1):
        ce_total_loss = 0.0
        model.train()
        time1 = time()
        n_batch = len(data_loader_train)

        for iter, batch in enumerate(data_loader_train):
            time2 = time()
            batch = [x.to(device) for x in batch]
            # y_true, y_postive = batch
            # [a,b,c],[a_h,b_h,c_h] => [a,a_h,b,b_h,c,c_h]
            # inputs = torch.stack([x.to(device) for x in inputs for n in range(2)])
            # inputs = torch.stack(list(chain.from_iterable(zip(y_true, y_postive)))).to(device)
            optimizer.zero_grad()

            ce_loss = model(batch, mode='train')

            if np.isnan(ce_loss.cpu().detach().numpy()):
                logging.info('ERROR (Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()
            
            ## update
            ce_loss.backward()
            optimizer.step()
            ce_total_loss += ce_loss.item()

        ## evaluate
        if (epoch % args.train_cfg.testing_epoch) == 0 or epoch == args.train_cfg.n_epochs:
            time3 = time()
            labels_pred_train, metrics_dict_train = evaluate(model, data_loader_train, device)
            labels_pred_valid, metrics_dict_valid = evaluate(model, data_loader_vali, device)
            labels_pred_test, metrics_dict_test = evaluate(model, data_loader_test, device)

            logging.info('Evaluation: Epoch {:04d} | Time {:.1f}s | Accuracy {:.3f}/{:.3f}/{:.3f} | F1-score {:.3f}/{:.3f}/{:.3f}'.format(epoch, time() - time0, metrics_dict_train['acc'], metrics_dict_valid['acc'], metrics_dict_test['acc'], metrics_dict_train['f1'], metrics_dict_valid['f1'], metrics_dict_test['f1']))
            epoch_list.append(int(epoch))
            metrics_list['acc'].append(metrics_dict_test['acc'])
            metrics_list['f1'].append(metrics_dict_test['f1'])
            metrics_list['confus_matrix'].append(metrics_dict_test['confus_matrix'])

            best_acc, should_stop = early_stopping(metrics_list['acc'], args.train_cfg.early_stopping_epoch, criterion=max)

            if should_stop and args.train_cfg.early_stopping_epoch != 0:
                break

            if metrics_list['acc'].index(best_acc) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch
    # save 
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx','Accuracy','F1-score','Confusion_matrix']
    metrics_df.append(metrics_list['acc'])
    metrics_df.append(metrics_list['f1'])
    metrics_df.append(metrics_list['confus_matrix'])

    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)
    
    pass

def evaluate(model, dataloader, device, model_file=None, load_self=False):
    model.eval()
    if model_file:
        load_model(model_file, load_self=load_self)
        model = model.to(device)

    labels_pred = []
    labels = []

    metrics_dict = {'acc': 0,'f1': 0, 'confus_matrix': 0}

    for i, batch in enumerate(dataloader):
        batch = map(lambda x: x.to(device), batch)
        with torch.no_grad():
            embed, label = batch 
            label_pred = model(embed,  mode='predict')
            labels_pred.append(label_pred)
            labels.append(label)
    # 预处理
    labels_pred = torch.cat(labels_pred, 0)
    labels = torch.cat(labels, 0)
    # 丢到cpu
    labels_pred = labels_pred.cpu().numpy()
    labels_pred = np.argmax(labels_pred, 1)
    labels = labels.cpu().numpy()
    # calc metrics
    acc = calc_acc(labels_pred, labels)
    f1 = calc_f1(labels_pred, labels)
    confus_matric = calc_confus_matric(labels_pred, labels)
    metrics_dict['acc'] = (acc)
    metrics_dict['f1'] = (f1)
    metrics_dict['confus_matrix'] = (confus_matric)

    return labels_pred, metrics_dict

if __name__ == "__main__":
    args = parse_args()
    train(args)