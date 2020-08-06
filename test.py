# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:00:31 2019

@author: Administrator
"""

import torch
import torch.nn as nn
import os
import show3d
import torchviz
import dataset
import sur3dnet
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description='Arguments for training', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_dir', type=str, default=R'H:\Sur3dNet\GpmmTrain', help='Dataset directory')
    parser.add_argument('--load_filename', type=str, default='sur3dnet_face_0309.pt')
    parser.add_argument('--rmap_filename', type=str, default=None, help='Remap file in loading checkpoint')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=3000)
    parser.add_argument('--point_num_max', type=int, default=24576)
    parser.add_argument('--print_model', type=bool, default=False)
    parser.add_argument('--print_param', type=bool, default=False)
    parser.add_argument('--test_graph', type=bool, default=False)
    parser.add_argument('--test_feat', type=bool, default=False)
    parser.add_argument('--test_plot3d', type=bool, default=True)

    return parser.parse_args()

if __name__=='__main__':
    print('Pytorch version:', torch.__version__)
    print('GPU available:', torch.cuda.device_count())
    
    args = parse_args()
    
    data_dir = args.data_dir
    load_filename = args.load_filename
    rmap_filename = args.rmap_filename
    batch_size = args.batch_size
    num_classes = args.num_classes
    point_num_max = args.point_num_max
         
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
        
    if args.print_model:
        print(model)
        
    if args.print_param:
        print('Params to learn:')
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print('\t', name)
    
    # Create test datasets
    test_datasets = {x:dataset.BcDataset(os.path.join(data_dir, x), num_classes, point_num_max=point_num_max, use_transform=False) for x in ['train']}
    
    # Test the model
    x = torch.stack(tuple(test_datasets['train'][i][0] for i in range(batch_size)), 0)
    x = x.to(device)
    
    if args.test_graph:
        model = model.train()
        torchviz.make_dot(model(x), params=dict(model.named_parameters())).view()
    
    if args.test_feat:
        model = model.eval()
        with torch.set_grad_enabled(False):
            fea = model(x, otype='feat')
        print('Feat:', fea[:, ::10])
    
    if args.test_plot3d:
        model = model.eval()
        with torch.set_grad_enabled(False):
            out = model(x, otype='image')
        show3dx = out.cpu().detach()
        show3d.show_scatter(show3dx[0, 0, :, 0], show3dx[0, 1, :, 0], show3dx[0, 2, :, 0], show3dx[0, x.size(1):x.size(1)+1, :, 0])
        
            