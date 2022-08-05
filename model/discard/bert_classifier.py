import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bert import BERT

class BERTClassifier(nn.Module):

    def __init__(self, bert_cfg, classifier=None, frozen_bert=False):
        super().__init__()
        self.bert = BERT(bert_cfg)
        if frozen_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.classifier = classifier
        
    
    def forward(self, *input, mode):
        if mode == 'train':
            return self.calc_ce_loss(*input)
        if mode == 'predict':
            return self.predict(*input)

    def predict(self, input_seqs, training=False): #, training
        h = self.bert(input_seqs)
        h = self.classifier(h, training)
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)
    
    def calc_ce_loss(self, batch):
        inputs, label = batch
        logits = self.predict(inputs, True)
        loss = self.criterion(logits, label) 
        return loss
