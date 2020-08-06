# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:38:51 2019

@author: Administrator
"""

import torch
import yi
import math

def knn_kdtree(x, y, k):
    
    # x: torch.FloatTensor[B, M, 3]
    # y: torch.FloatTensor[B, N, 3]
    # output: tuple(torch.FloatTensor[B, N, k], torch.LongTensor[B, N, k])
    print(y[:, 752, :])
    output = yi.knn(x.cpu(), y.contiguous(), k)
    print(y[:, 752, :], output[1][:, 752, :])
    
    return (output[0], output[1].long())

def knn_linear(x, y, k):

    # x: torch.FloatTensor[B, M, 3]
    # y: torch.FloatTensor[B, N, 3]
    # output: tuple(torch.FloatTensor[B, N, k], torch.LongTensor[B, N, k])
    
    inner = -2 * torch.matmul(y, x.transpose(1, 2))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    yy = torch.sum(y**2, dim=2, keepdim=True)

    pairwise_distance = yy + inner + xx.transpose(1, 2)
    
    output = pairwise_distance.topk(k=k, dim=-1, largest=False)
    
    return output

def node_sample_kdtree(x, y, n, node_radius=3, node_num=8):
    
    # x: torch.FloatTensor[B, M, 3]
    # y: torch.FloatTensor[B, N, 3]
    # n: torch.FloatTensor[B, N, 3]
    # knn: torch.LongTensor[B, N, node_num]
    
    # Local coordinate
    # u: torch.FloatTensor[B, N, 3]
    # v: torch.FloatTensor[B, N, 3]
    # uv: torch.FloatTensor[2, B, N, 3]
    print(n[:, 47, :])
    
    u = torch.stack((1 - n[:, :, 0] ** 2, -n[:, :, 0] * n[:, :, 1], -n[:, :, 0] * n[:, :, 2]), 2)
    v = torch.stack((torch.zeros(n.size(0), n.size(1)).to(n), n[:, :, 2], -n[:, :, 1]), 2)
    
    
    
    uv = torch.stack((u, v), 0) / torch.sqrt(u[:, :, 0:1].unsqueeze(0))
    print(torch.sqrt(u[:, 47, 0:1].unsqueeze(0)))
    
    #nvtab: torch.FloatTensor[2, 1, 1, node_num, 1]
    tab = torch.arange(0, 2 * math.pi, 2 * math.pi / node_num)
    uvtab = torch.stack((torch.cos(tab), torch.sin(tab)), 0).view(2, 1, 1, node_num, 1).to(n)
    
    # Get nodes
    # node: torch.FloatTensor[B, N, 8, 3]
    print(uv[:, :, 47, :])
    print(n[:, 47, :])
    node = y.unsqueeze(2).repeat(1, 1, node_num, 1) + (uv.unsqueeze(3) * uvtab).sum(0) * node_radius
    print(node[:, 47, 0, :])
    node = node.view(node.size(0), -1, node.size(3))
    print(node[:, 752, :])
    _, knn = knn_kdtree(x, node, 1)
    knn = knn.view(knn.size(0), -1, node_num)
    
    return knn

def node_sample_linear(x, y, n, node_radius=3, node_num=8):
    
    # x: torch.FloatTensor[B, M, 3]
    # y: torch.FloatTensor[B, N, 3]
    # n: torch.FloatTensor[B, N, 3]
    # knn: torch.LongTensor[B, N, node_num]
    
    # Local coordinate
    # u: torch.FloatTensor[B, N, 3]
    # v: torch.FloatTensor[B, N, 3]
    # uv: torch.FloatTensor[2, B, N, 3]
    u = torch.stack((1 - n[:, :, 0] ** 2, -n[:, :, 0] * n[:, :, 1], -n[:, :, 0] * n[:, :, 2]), 2)
    v = torch.stack((torch.zeros(n.size(0), n.size(1)).to(n), n[:, :, 2], -n[:, :, 1]), 2)
    uv = torch.stack((u, v), 0) / torch.sqrt(u[:, :, 0:1].unsqueeze(0))
    
    #nvtab: torch.FloatTensor[2, 1, 1, node_num, 1]
    tab = torch.arange(0, 2 * math.pi, 2 * math.pi / node_num)
    uvtab = torch.stack((torch.cos(tab), torch.sin(tab)), 0).view(2, 1, 1, node_num, 1).to(n)
    
    # Get nodes
    # node: torch.FloatTensor[B, N, 8, 3]
    node = (uv.unsqueeze(3) * uvtab).sum(0) * node_radius
    
    _, knn = knn_linear(x, y, min(49, x.size(1)))
    ynn = torch.gather(x, 1, knn.view(knn.size(0), -1, 1).repeat(1, 1, y.size(2))).view(x.size(0), knn.size(1), knn.size(2), x.size(2))
    print(ynn.size())
    ry = (ynn - y.unsqueeze(2))# * (1 - n.unsqueeze(2))
    print(ry.size(), node.size())
    d = (ry.unsqueeze(2) - node.unsqueeze(3)).norm(dim=4)
    print(d.size())
    _, idx = d.min(dim=3)
    knn = torch.gather(knn, 2, idx)
    
    return knn

def node_sample(x, y, n, node_radius=3, node_num=8, ram_limit=256):
    
    # x: torch.FloatTensor[B, M, 3]
    # y: torch.FloatTensor[B, N, 3]
    # n: torch.FloatTensor[B, N, 3]
    # knn: torch.LongTensor[B, N, node_num]
    
    data_vol = x.size(1) * y.size(1)
    cost_kdtree = 0.29 + 1.5e-9 * data_vol
    cost_linear = 0.03 + 9.7e-9 * data_vol
    
    # ram comsumption on knn_linear (MB)
    ram_linear = data_vol * 4 / 1024 / 1024
    
#    print(cost_kdtree, cost_linear, ram_linear)
    
    if ram_linear > ram_limit or cost_kdtree < cost_linear:
#        print('K', end='')
        knn = node_sample_kdtree(x, y, n, node_radius, node_num)
    else:
#        print('L', end='')
        batch_num = x.size(0)
        batch_per = int(ram_limit // ram_linear)
        batch_fst = int((batch_num - 1) % batch_per + 1)
        batch_itr = int((batch_num - batch_fst) // batch_per)
        knn = node_sample_kdtree(x[0:batch_fst, :, :], y[0:batch_fst, :, :], n[0:batch_fst, :, :], node_radius, node_num)    
        for i in range(batch_itr):
            bs = batch_fst + i * batch_per
            be = batch_fst + (i + 1) * batch_per
            knn = torch.cat((knn, node_sample_linear(x[bs:be, :, :], y[bs:be, :, :], n[bs:be, :, :], node_radius, node_num)), 0) 
    
    return knn

