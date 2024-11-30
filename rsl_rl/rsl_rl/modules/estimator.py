from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm

# 估计器
class Estimator(nn.Module):
    """
    这个估计器方法是一个多层感知机(MLP), 用于根据输入特征估计某些输出
    """
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu",
                        **kwargs):
        """
        初始化估计器
        :param input_dim: 输入特征的维度
        :param output_dim: 输出特征的维度
        :param hidden_dims: 隐藏层的维度
        """
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # 激活函数
        activation = get_activation(activation)
        estimator_layers = []
        # 添加输入层及激活函数
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                # 添加输出层
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                # 添加隐藏层及激活函数
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        # 将所有层组合成一个网络
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)

class Discriminator(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        # self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # init_weight(self.hidden1)
        # self.hidden1.bias.data.zero_()
        # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # init_weight(self.hidden2)
        # self.hidden2.bias.data.zero_()
        # self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        # init_weight(self.q, initializer="xavier uniform")
        # self.q.bias.data.zero_()

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)

class DiscriminatorLSD(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorLSD, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(spectral_norm(nn.Linear(n_states, hidden_dims[0])))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], n_skills)))
            else:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], hidden_dims[l + 1])))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)
        
class DiscriminatorContDIAYN(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorContDIAYN, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)

    def forward(self, states):
        return torch.nn.functional.normalize(self.discriminator(states))

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)