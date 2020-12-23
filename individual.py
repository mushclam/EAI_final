import torch
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
import utils

from customized_model import CustomizedModel

class Individual:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.size = in_features * out_features
        self.distance = 0

    def initialization(self, mask=None):
        if mask is None:
            self.mask = torch.randint(2, (self.out_features, self.in_features)).float()
        else:
            if isinstance(mask, torch.Tensor):
                self.mask = mask
            else:
                self.mask = torch.Tensor(mask)

    def getZeros(self):
        n_zeros = self.size - self.mask.count_nonzero().item()
        return n_zeros

    def evaluation(self, model, classes, dataloader, device):
        # Copy model
        tmp_model = utils.copy_model(model)
        # Load mask to device
        # self.mask.to(device)
        tmp_model[3].weight = nn.Parameter(tmp_model[3].weight * self.mask)
        tmp_model.to(device)
        # Evaluation
        accuracy = utils.eval(tmp_model, classes, dataloader, device)
        self.fitness = [
            self.getZeros(),
            accuracy
        ]

        return self.fitness

    def to(self, device):
        self.model.to(device)

    def dominate(self, individual):
        for i in range(len(self.fitness)):
            if (self.fitness[i] < individual.fitness[i]):
                return False
        return True
