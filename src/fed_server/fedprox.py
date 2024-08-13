from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.fed_client.fedprox_client import FedProx_CLient
from src.optimizers.gd import GD
import numpy as np
import torch
import time
from src.fed_client.base_client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy

import torch.nn.functional as F
from src.utils.tools import Metrics
class FedProxTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power,g_N0):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'])
        super(FedProxTrainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, g_N0,model, self.optimizer)

    def train(self):
        print('>>> Select {} clients per round \n'.format(int(self.per_round_c_fraction * self.clients_num)))

        # Fetch latest flat model parameter
        # self.latest_global_model = self.get_model_parameters()
        print("init", self.latest_global_model['fc2.bias'])
        for round_i in range(self.num_round):
            # self.w_last_global = copy.deepcopy(self.latest_global_model)
            # print("{}, {}".format(round_i, self.latest_global_model))
            # Test latest model on train data
            # self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_testdata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients()

            # Solve minimization locally
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)


            # comp cost
            # self.comptuing_delay_energy(selected_clients)
            self.metrics.update_cost(round_i, self.cost.delay_Sum, self.cost.energy_Sum)

            # Track communication cost
            self.metrics.extend_communication_stats(round_i, stats)

            # Update latest model
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            # self.optimizer.adjust_learning_rate(round_i)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train datap
        # self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_testdata(self.num_round)

        # # Save tracked information
        self.metrics.write()

    def select_clients(self):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        index = np.random.choice(len(self.clients), num_clients, replace=False,)
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        return select_clients

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        all_client = []
        for i in range(len(clients_label)):
            local_client = FedProx_CLient(self.options, i, self.clients_attr, TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.model, self.optimizer)
            all_client.append(local_client)

        return all_client
    
    def aggregate_parameters(self, solns, **kwargs):


        averaged_solution = torch.zeros_like(self.latest_global_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        self.simple_average = False
        if self.simple_average:
            num = 0
            for num_sample, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            num = 0
            for num_sample, local_solution in solns:
                # print(local_solution)
                num += num_sample
                averaged_solution += num_sample * local_solution
            averaged_solution /= num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()
    

    def aggregate_parameters(self, local_model_paras_set):

        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = 0
        num = 1
        for var in averaged_paras:
            averaged_paras[var] = 0
        for num_sample, local_model_paras in local_model_paras_set:
            for var in averaged_paras:
                averaged_paras[var] += local_model_paras[var]
            train_data_num += num_sample
            num += 1
        for var in averaged_paras:
            averaged_paras[var] /= num

        return averaged_paras
