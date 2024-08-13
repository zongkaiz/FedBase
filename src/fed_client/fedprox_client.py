from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
criterion = F.cross_entropy
mse_loss = nn.MSELoss()
from .base_client import Client


class FedProx_CLient(Client):
    def __init__(self, options, id, model, optimizer, local_dataset, system_attr,  ):

        super(FedProx_CLient, self).__init__(options, id, model, optimizer, local_dataset, system_attr, )

        self.global_model_parameters = None
        self.mu = 0.01

    def local_train(self, ):
        bytes_w = self.model_bytes
        begin_time = time.time()
        self.global_model_parameters = None
        self.global_model_parameters = copy.deepcopy(self.get_model_parameters()) 
        local_model_paras, dict = self.local_update(self.local_dataset, self.options, )
        # print("sb", self.get_model_parameters()['fc2.bias'])
        end_time = time.time()
        bytes_r = self.model_bytes
        stats = {'id': self.id, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats


    def local_update(self, local_dataset, options, ):
        localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        print("sb", self.get_model_parameters()['fc2.bias'])
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                self.optimizer.zero_grad()
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(self.get_model_parameters().values(), self.global_model_parameters.values()):
                    proximal_term += torch.norm(w - w_t, p=2).item()
                loss = criterion(pred, y) + (self.mu / 2) * proximal_term
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        local_model_paras = copy.deepcopy(self.get_model_parameters())
        print("sc", self.get_model_parameters()['fc2.bias'])
        comp = self.options['local_epoch'] * train_total * self.flops
        return_dict = {"id": self.id,
                        "comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict
        
