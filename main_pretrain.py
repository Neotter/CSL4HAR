import sys
import random
from itertools import chain
from time import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from dataloader.pretrain import DatasetSelector4Pretrain

from model.bert4cl import *
from model.decoder_projhead import ProjectHead
from praser.csl4har import *
from trainer.loss_fn import info_nce_loss
from utils.log_helper import *
from utils.get_device import *
from utils.model_helper import *
from torch.utils.tensorboard import SummaryWriter 

def pretrain(args):
    # log_path
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # tensorboard
    if args.enable_tensor_board == True:
        writer = SummaryWriter(args.save_dir)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    dataloader = DatasetSelector4Pretrain(args)
    data_loader_train, data_loader_vali = dataloader.get_dataloader()
    
    logging.info('Train Size: {}, Vali Size: {}'.format(data_loader_train.dataset.__len__(), data_loader_vali.dataset.__len__()))

    # construct model & optimizer
    input_dim = dataloader.dataset_cfg.dimension
    embed_dim = args.encoder_cfg.embed_dim
    output_dim = args.encoder_cfg.embed_dim

    if args.augmentation == 'clipping' or args.augmentation == 'delwords' or args.augmentation == 'spanmasking':
        decoder = ProjectHead(args, embed_dim, output_dim, meanPooling=True)
    else:
        decoder = ProjectHead(args, embed_dim, output_dim, meanPooling=False)

    model = BERT4CL(args.encoder_cfg, input_dim, embed_dim, info_nce_loss, following_model=decoder)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.train_cfg.lr)

    logging.info(model)
    logging.info(optimizer)

    # initialize metrics
    best_epoch = -1
    best_loss = 0
    epoch_list = []
    metrics_list = {'nce_loss': []}

    # train
    time0 = time()
    for epoch in range(1, args.train_cfg.n_epochs+1):
        nce_total_loss = 0.0
        model.train()
        time1 = time()
        n_batch = len(data_loader_train)

        for iter, batch in enumerate(data_loader_train):
            time2 = time()
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad()
            nce_loss = model(batch, mode='pretrain')

            if np.isnan(nce_loss.cpu().detach().numpy()):
                logging.info('ERROR (Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()
            
            ## update
            nce_loss.backward()
            optimizer.step()
            nce_total_loss += nce_loss.item()

        ## evaluate
        if (epoch % args.train_cfg.testing_epoch) == 0 or epoch == args.train_cfg.n_epochs:
            time3 = time()
            loss_test = evaluate(model, data_loader_vali, device)
            if args.enable_tensor_board == True:
                writer.add_scalar('NCE_loss_test', loss_test, epoch)
                writer.add_scalar('NCE_loss_training', nce_total_loss / n_batch, epoch)
            # epoch | epoch_time / total_time | Iter_mean_loss/test_loss | 
            logging.info('Evaluation: Epoch {:04d} | Time {:.1f}/{:.1f}s | Loss {:.4f}/{:.4f}'.format(epoch, time3-time1, time() - time0, nce_total_loss / n_batch, loss_test))
            epoch_list.append(epoch)
            metrics_list['nce_loss'].append(loss_test)
            best_loss, should_stop = early_stopping(metrics_list['nce_loss'], args.train_cfg.early_stopping_epoch)
            if should_stop and args.train_cfg.early_stopping_epoch != 0:
                break
            if args.train_cfg.saving_epoch != 0 and epoch % args.train_cfg.saving_epoch == 0 :
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                save_model(model, args.save_dir, epoch)

            if metrics_list['nce_loss'].index(best_loss) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save 
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx','nce_loss']
    metrics_df.append(metrics_list['nce_loss'])
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best Evaluation: Epoch {:04d} | Total Time {:.1f}s | infoNCE LOSS [{:.4f}]'.format(
        int(best_metrics['epoch_idx']), time()-time0 , best_metrics['nce_loss']))
    pass

def evaluate(model, dataloader, device, model_file=None, load_self=False):
    model.eval()
    if model_file:
        load_model(model_file, load_self=load_self)
        model = model.to(device)
    results = []
    for i, batch in enumerate(dataloader):
        # batch = map(lambda x: x.to(device), batch)
        with torch.no_grad():  
            batch = [x.to(device) for x in batch]       
            nce_loss = model(batch, mode='pretrain')
            results.append(nce_loss)
    return torch.tensor(results).mean().cpu().numpy()

if __name__ == "__main__":
    args = parse_args('pretrain')
    # args.mask_cfg = args.mask_cfg._replace(mask_ratio=0.5)
    pretrain(args)