import os
from collections import OrderedDict

import torch

def early_stopping(score_list, stopping_steps, criterion=min):
    best_score = criterion(score_list)
    best_step = score_list.index(best_score)
    if len(score_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_score, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            # os.system('rm {}'.format(old_model_state_file))
            os.remove(old_model_state_file)


def load_model(model, model_file, load_self=False):
    """ load saved model or pretrained transformer (a part of model) """
    if model_file:
        print('Loading the model from', model_file)
        if load_self:
            model.load_self(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
            # print(checkpoint['model_state_dict'])
            checkpoint = checkpoint['model_state_dict']
            # for key, v in enumerate(checkpoint):
            #     print(key,v)
            model_dict = model.state_dict()
            checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
        return model

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)