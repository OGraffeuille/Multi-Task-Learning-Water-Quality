import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utility import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MTL architecture
# Task specific layers are effectively fully connected layers,
# a mask matrix is used during feedforward calculations to prevent
# connections between nodes of different tasks
class MTL_Net(nn.Module):

    def __init__(self, num_inputs, shared_layer_sizes, task_layer_sizes, num_tasks):
        super(MTL_Net, self).__init__()

        self.num_inputs = num_inputs
        self.shared_layer_sizes = shared_layer_sizes
        self.task_layer_sizes = task_layer_sizes
        self.num_tasks = num_tasks
        self.num_outputs = self.task_layer_sizes[-1] if len(task_layer_sizes) > 0 else self.shared_layer_sizes[-1]
        self.is_MTL = len(self.task_layer_sizes) > 0

        prev_layer_size = num_inputs

        # Define shared Layers
        self.shared_layers = nn.ModuleList()
        for shared_layer_size in self.shared_layer_sizes:
            self.shared_layers.append(nn.Linear(prev_layer_size, shared_layer_size))
            prev_layer_size = shared_layer_size

        # Define task-specific Layers
        if self.is_MTL:
            assert len(self.task_layer_sizes) < 2, "ERROR: only one task-specific layer implemented"
            task_layer_size = self.task_layer_sizes[0] #always 1 for regression
            self.task_layer = nn.Linear(prev_layer_size, task_layer_size * num_tasks)


    def forward(self, x):

        for i in range(len(self.shared_layers)):
            x = self.shared_layers[i](x)
            if self.is_MTL or (i < len(self.shared_layers)-1):
                x = F.relu(x)

        if self.is_MTL:
            x = self.task_layer(x)

        return x


class MDN_Net(nn.Module):

    def __init__(self, num_inputs, shared_layer_sizes, task_layer_sizes, num_tasks, num_gaussians):
        super(MDN_Net, self).__init__()

        self.num_inputs = num_inputs
        self.shared_layer_sizes = shared_layer_sizes
        self.task_layer_sizes = task_layer_sizes
        self.num_tasks = num_tasks
        self.num_outputs = self.task_layer_sizes[-1] if len(task_layer_sizes) > 0 else self.shared_layer_sizes[-1]
        self.num_gaussians = num_gaussians
        self.is_MTL = len(self.task_layer_sizes) > 0

        prev_layer_size = num_inputs

        # Define shared Layers
        self.shared_layers = nn.ModuleList()
        for l in range(len(self.shared_layer_sizes)):
            shared_layer_size = self.shared_layer_sizes[l]
            if l == len(self.shared_layer_sizes) - 1:
                shared_layer_size *= self.num_gaussians
            self.shared_layers.append(nn.Linear(prev_layer_size, shared_layer_size))
            prev_layer_size = shared_layer_size

        # Define task-specific Layers
        if self.is_MTL:
            assert len(self.task_layer_sizes) < 2, "ERROR: only one task-specific layer implemented"
            task_layer_size = self.task_layer_sizes[0]  # always 1 for regression
            self.task_layer = nn.Linear(prev_layer_size, task_layer_size * num_tasks * num_gaussians)
            print(task_layer_size * num_tasks * num_gaussians)


    def forward(self, x):
        for i in range(len(self.shared_layers)):
            x = self.shared_layers[i](x)
            if self.is_MTL or (i < len(self.shared_layers) - 1):
                x = F.relu(x)

        if self.is_MTL:
            x = self.task_layer(x)
            x = x.view(-1, self.num_tasks, self.task_layer_sizes[0] * self.num_gaussians)
            
        x_c = x.clone()
        m = self.num_gaussians
        x_c[..., m:2*m] = torch.exp(x[..., m:2*m])            # sigma
        x_c[..., 2*m:] = F.softmax(x[..., 2*m:], dim=-1)      # pi

        return x_c