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
from praser.pretrain import *
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
    embed_dim = args.model_cfg.embed_dim
    output_dim = args.dataset_cfg.dimension

    decoder = ProjectHead(args, embed_dim, embed_dim)
    model = BERT4CL(args.model_cfg, input_dim, embed_dim, following_model=decoder)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.train_cfg.lr)

    logging.info(model)
    logging.info(optimizer)

    # initialize metrics
    best_epoch = -1
    best_loss = 0
    epoch_list = []
    metrics_list = {'nce_loss': []}

    time0 = time()
    for epoch in range(1, args.train_cfg.n_epochs+1):
        nce_total_loss = 0.0
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
            writer.add_scalar('NCE_loss_test', loss_test, epoch)
            writer.add_scalar('NCE_loss_training', nce_total_loss / n_batch, epoch)
            logging.info('Evaluation: Epoch {:04d} | Time {:.1f}s | Iter Mean Loss {:.4f} | Test Loss {:.4f}'.format(epoch, time() - time0, nce_total_loss / n_batch, loss_test))
            epoch_list.append(epoch)
            metrics_list['nce_loss'].append(loss_test)
            best_loss, should_stop = early_stopping(metrics_list['nce_loss'], args.train_cfg.early_stopping_epoch)
            if should_stop and args.train_cfg.early_stopping_epoch != 0:
                break
            # if epoch % args.train_cfg.saving_epoch == 0:
            #     print(epoch % args.train_cfg.saving_epoch)
            #     save_model(model, args.save_dir, epoch)
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
    logging.info('Best Evaluation: Epoch {:04d} | Time {:.1f}s | NCE LOSS [{:.4f}]'.format(
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
            # inputs, label = batch
            # [a,b,c] => [a,a,b,b,c,c]
            # inputs = torch.stack([x.to(device) for x in inputs for n in range(2)])     
            batch = [x.to(device) for x in batch]       
            nce_loss = model(batch, mode='pretrain')
            results.append(nce_loss)
    return torch.tensor(results).mean().cpu().numpy()

if __name__ == "__main__":
    args = parse_args()
    pretrain(args)