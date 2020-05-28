import torch
import copy
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def train_net(net, num_epoches, lr, img_noisy, img_clean, k_channels, net_input=None, lr_decay_step=0):
    if net_input == None:
        net_input = torch.zeros([1, k_channels, 16, 16])
        net_input = net_input.uniform_()
        net_input *= 1./10

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if lr_decay_step != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    else:
        scheduler = None

    loss_noisy = []
    loss_clean = []

    best_net = copy.deepcopy(net)
    best_net_loss = 1000000

    net = net.cuda()
    net_input = net_input.cuda()
    img_noisy = img_noisy.cuda()
    img_clean = img_clean.cuda()
    net.train()

    for epoch in range(num_epoches):
        optimizer.zero_grad()
        output = net(net_input)
        loss = loss_function(output, img_noisy)
        # print(loss)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        loss_noisy.append(loss.item())
        loss2 = loss_function(output, img_clean)
        loss_clean.append(loss2.item())
        
        if loss < best_net_loss:
            best_net = copy.deepcopy(net)
            best_net_loss = loss

        if epoch % 1000 == 0:
            print(loss)            
    
    best_net.eval()

    return best_net, loss_noisy, loss_clean, net_input
