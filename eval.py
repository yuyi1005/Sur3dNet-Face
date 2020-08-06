# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:47:51 2019

@author: Administrator
"""

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import dataset
import sur3dnet
import argparse

def eval_cmc_roc(model, dataloaders, device):

    model = model.eval()
    
    gallenum = len(dataloaders['gallery'].dataset.filelist)
    probenum = len(dataloaders['probe'].dataset.filelist)

    galle = torch.zeros(gallenum, 1, model.module.fc.in_features).to(device)
    galab = torch.zeros(gallenum, 1, dtype=torch.long).to(device)
    
    idx = 0
    itera_cnt = 0
    # Iterate over data.
    for inputs, labels in dataloaders['gallery']:
    
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            feat = model(inputs, otype='feat').detach()
            
        for i in range(labels.size(0)):
            galle[idx, 0, :] = feat[i, :]
            galab[idx, 0] = labels[i]
            idx += 1
        itera_cnt += 1
        if itera_cnt % 10 == 0:
            print('.', end='', flush=True)
        if itera_cnt % 100 == 0:
            print('({})'.format(itera_cnt), end='')
            
    print()
    
    probe = torch.zeros(1, probenum, model.module.fc.in_features).to(device)
    prlab = torch.zeros(1, probenum, dtype=torch.long).to(device)
    
    idx = 0
    itera_cnt = 0
    # Iterate over data.
    for inputs, labels in dataloaders['probe']:
    
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            feat = model(inputs, otype='feat').detach()
            
        for i in range(labels.size(0)):
            probe[0, idx, :] = feat[i, :]
            prlab[0, idx] = labels[i]
            idx += 1
        itera_cnt += 1
        if itera_cnt % 10 == 0:
            print('.', end='', flush=True)
        if itera_cnt % 100 == 0:
            print('({})'.format(itera_cnt), end='')
            
    print()
    
    galle /= galle.norm(dim=2, keepdim=True)
    probe /= probe.norm(dim=2, keepdim=True)
    distan = (galle * probe).sum(2)
    idsame = galab == prlab
#    print(distan)
#    print(idsame)
    
    # CMC curve
    rrn = []
    _, idx = torch.sort(distan, 0, descending=True)
    outlab = torch.gather(galab.repeat(1, idx.size(1)), 0, idx)
    rankrate = torch.sum(outlab == prlab, 1).float() / probenum
    ranknum = min(rankrate.size(0), 19)
    for i in range(ranknum):
        if i == 0:
            rrn.append(rankrate[i].item())
        else:
            rrn.append(rrn[i - 1] + rankrate[i].item())
    
    plt.subplot(1, 2, 1)
    plt.xticks(range(ranknum + 1) if ranknum < 10 else range(1, ranknum + 1, 2))
    plt.plot(range(1, ranknum + 1), rrn)
    plt.xlim(1, ranknum)
    plt.grid(True)
    
    # ROC curve
    tpr = []
    fpr = []
    for i in range(200):
        thres = 1 - 0.005 * i
        pos = (distan >= thres)
        tpr.append(((pos * idsame).sum().float() / idsame.sum()).item())
        fpr.append(((pos * (~idsame)).sum().float() / (~idsame).sum()).item())
    
    vr3 = 0
    auc = tpr[0] * fpr[0] + tpr[len(fpr) - 1] * (1 - fpr[len(fpr) - 1])
    for i in range(len(fpr) - 1):
        if fpr[i] <= 1e-3 and fpr[i + 1] >= 1e-3:
            if fpr[i + 1] - fpr[i] == 0:
                vr3 = (tpr[i] + tpr[i + 1]) / 2
            else:
                vr3 = (tpr[i] * (fpr[i + 1] - 1e-3) + tpr[i + 1] * (1e-3 - fpr[i])) / (fpr[i + 1] - fpr[i])
        auc += (tpr[i + 1] + tpr[i]) / 2 * (fpr[i + 1] - fpr[i])
        
    print('RR1: {:.4f} VR(FAR=1e-3): {:.4f} AUC: {:.4f}'.format(rrn[0], vr3, auc))
    
    xmin = 1e-4
    for i in range(200):
        if fpr[i] < xmin:
            fpr[i] = xmin
            
    plt.subplot(1, 2, 2)
    plt.xscale('log', nonposx='clip')
    plt.plot(fpr, tpr)
    plt.xlim(xmin, 1)
    plt.grid(True)
    plt.show()

    return 0

def parse_args():
    
    parser = argparse.ArgumentParser(description='Arguments for training', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_dir', type=str, default=R'H:\Sur3dNet\BosphorusEval', help='Dataset directory')
    parser.add_argument('--data_gallery', type=str, default='gallery', help='Gallery folder name')
    parser.add_argument('--data_probe', type=str, default='E', help='Probe folder name')
    parser.add_argument('--load_filename', type=str, default='sur3dnet_face_0309.pt')
    parser.add_argument('--rmap_filename', type=str, default=None, help='Remap file in loading checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=3000)
    parser.add_argument('--point_num_max', type=int, default=24576)

    return parser.parse_args()

if __name__ == '__main__':
    print('Pytorch version:', torch.__version__)
    print('GPU available:', torch.cuda.device_count())
    
    args = parse_args()
    
    data_dir = args.data_dir
    load_filename = args.load_filename
    rmap_filename = args.rmap_filename
    batch_size = args.batch_size
    num_classes = args.num_classes
    point_num_max = args.point_num_max
    data_name = {'gallery':args.data_gallery, 'probe':args.data_probe}
         
    # Detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Create the model and send the model to GPU
    model = sur3dnet.Sur3dNet(num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device)
    
    if load_filename is not None:
        # Load the model from file
        model_dict = model.state_dict()
        saved_dict = torch.load(load_filename)
            
        if rmap_filename is not None:
            with open(rmap_filename, mode='r') as f:
                loadmap = f.readlines()
            for s in loadmap:
                k = s.split()
                saved_dict[k[1]] = saved_dict.pop(k[0])
        
        model_dict.update(saved_dict)
        model.load_state_dict(model_dict)
    
    # Create test datasets, as well as dataloaders
    eval_datasets = {x:dataset.BcDataset(os.path.join(data_dir, data_name[x]), num_classes, point_num_max=point_num_max) for x in ['gallery', 'probe']}
    eval_dataloaders = {x:torch.utils.data.DataLoader(eval_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) for x in ['gallery', 'probe']}
    print('Num gallery/probe: ', len(eval_datasets['gallery']), len(eval_datasets['probe']))
    
    # Plot CMC and ROC curves
    eval_cmc_roc(model, eval_dataloaders, device)
        
        