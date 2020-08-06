
import torch
import torch.nn as nn
from collections import OrderedDict
import random
import sampling
import yi
import angle_linear
    
class Mlp(nn.Module):
    
    def __init__(self, planes_tab=[], learnable=True):
        super(Mlp, self).__init__()
        
        self.layers = self.make_layer(planes_tab)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if not learnable:
            for p in self.parameters():
                p.requires_grad = False
                
    def make_layer(self, planes_tab):
        
        layers = nn.ModuleList()
        for i in range(len(planes_tab) - 1):
            layers.append(nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(planes_tab[i], planes_tab[i + 1], kernel_size=(1, 1), bias=False)), 
                    ('bn', nn.BatchNorm2d(planes_tab[i + 1])), 
                    ('relu', nn.ReLU(inplace=True))])))

        return layers
    
    def forward(self, x):

        x = x.view(x.size(0), x.size(1), 1, -1)

        # layers in ModuleList
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        x, _ = x.max(dim=3, keepdim=True)
        
        return x
    
class PointNet(nn.Module):
    
    # xyzn_type=['xyz', 'xyzn', 'xyznc']
    # feat_type=['xyz', 'xyzn', 'xyznc', 'inner', 'none']
    # samp_type=['fps', 'dfps', 'rand']
    # quer_type=['ball', 'knn']
    def __init__(self, planes_tab=[], node_radius=3, node_num=32, downsample=4, 
                 xyzn_type='xyznc', feat_type='xyzn', samp_type='dfps', quer_type='ball', learnable=True):
        super(PointNet, self).__init__()
        
        self.node_radius = node_radius
        self.node_num = node_num
        self.downsample = downsample
        self.xyzn_type = xyzn_type
        self.feat_type = feat_type
        self.samp_type = samp_type
        self.quer_type = quer_type
        
        if self.xyzn_type == 'xyz':
            self.xyzn_size = 3
        elif self.xyzn_type == 'xyzn':
            self.xyzn_size = 6
        elif self.xyzn_type == 'xyznc':
            self.xyzn_size = 7
        
        if self.feat_type == 'xyz':
            planes_tab[0] += 3
        elif self.feat_type == 'xyzn':
            planes_tab[0] += 6
        elif self.feat_type == 'xyznc':
            planes_tab[0] += 7
        elif self.feat_type == 'inner':
            planes_tab[0] += 5
        
        self.layers = self.make_layer(planes_tab)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if not learnable:
            for p in self.parameters():
                p.requires_grad = False
                
    def make_layer(self, planes_tab):
        
        layers = nn.ModuleList()
        for i in range(len(planes_tab) - 1):
            layers.append(nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(planes_tab[i], planes_tab[i + 1], kernel_size=(1, 1), bias=False)), 
                    ('bn', nn.BatchNorm2d(planes_tab[i + 1])), 
                    ('relu', nn.ReLU(inplace=True))])))

        return layers
    
    def forward(self, x):
        
        if self.samp_type == 'fps':
            k = yi.furthest_point_sampling(x[:, 0:3, :, 0].transpose(1, 2).contiguous(), x.size(2) // self.downsample, 0).long()
            p = torch.gather(x[:, 0:self.xyzn_size, :, :], 2, k.view(k.size(0), 1, -1, 1).repeat(1, self.xyzn_size, 1, 1)).view(x.size(0), self.xyzn_size, k.size(1), 1)
        elif self.samp_type == 'dfps':
            if self.training:
                w = x[:, 6:7, :, 0].pow(random.uniform(-0.2, 0.2))
                w[x[:, 0:3, :, 0].norm(dim=1, keepdim=True) > 65 + random.uniform(-15, 15)] = 0
            else:
                w = x[:, 6:7, :, 0].pow(0.0)
                w[x[:, 0:3, :, 0].norm(dim=1, keepdim=True) > 65] = 0
            k = yi.furthest_point_sampling(torch.cat((x[:, 0:3, :, 0], w), 1).transpose(1, 2).contiguous(), x.size(2) // self.downsample, 0).long()
            p = torch.gather(x[:, 0:self.xyzn_size, :, :], 2, k.view(k.size(0), 1, -1, 1).repeat(1, self.xyzn_size, 1, 1)).view(x.size(0), self.xyzn_size, k.size(1), 1)
        elif self.samp_type == 'rand':
            p = x[:, 0:self.xyzn_size, ::self.downsample, :]
        
        if self.quer_type == 'ball':
            k = yi.ball_query(x[:, 0:3, :, 0].transpose(1, 2).contiguous(), p[:, 0:3, :, 0].transpose(1, 2).contiguous(), self.node_radius, self.node_num).long()
#            print(k.size(2), k.unique(dim=2).numel() / k.numel())
        elif self.quer_type == 'knn':
            _, k = sampling.knn_linear(x[:, 0:3, :, 0].transpose(1, 2), p[:, 0:3, :, 0].transpose(1, 2), self.node_num)
        
        if k.min().item() < 0:
            raise Exception('k < 0')
            
        if k.max().item() > x.size(2):
            raise Exception('k >= x.size(2)')
        
        v = torch.gather(x, 2, k.view(k.size(0), 1, -1, 1).repeat(1, x.size(1), 1, 1)).view(x.size(0), x.size(1), k.size(1), k.size(2))
        x = p.detach()
        
        # v[batch_num, channels, points, directions]
        # first xyzn_size channels for xyznc, others for previors value
        if self.feat_type == 'xyz':
            vx = v[:, 0:3, ...] - x[:, 0:3, ...]
            v = torch.cat((vx.detach(), v[:, self.xyzn_size:, ...]), 1)
        elif self.feat_type == 'xyzn':
            vx = v[:, 0:3, ...] - x[:, 0:3, ...]
            vn = v[:, 3:6, ...]
            v = torch.cat((vx.detach(), vn.detach(), v[:, self.xyzn_size:, ...]), 1)
        elif self.feat_type == 'xyznc':
            vx = v[:, 0:3, ...] - x[:, 0:3, ...]
            vn = v[:, 3:6, ...]
            cc = v[:, 6:7, ...]
            v = torch.cat((vx.detach(), vn.detach(), cc.detach(), v[:, self.xyzn_size:, ...]), 1)
        elif self.feat_type == 'none':
            v = v[:, self.xyzn_size:, ...]
        
        # layers in ModuleList
        for i in range(len(self.layers)):
            v = self.layers[i](v)
        
        v, _ = v.max(dim=3, keepdim=True)
        x = torch.cat((x, v), 1)
        
        return x
    
class Sur3dNet(nn.Module):
    
    def __init__(self, num_classes):
        super(Sur3dNet, self).__init__()
        
        self.pointnet1 = PointNet(planes_tab=[0, 32, 32, 32], node_radius=4, node_num=24, downsample=6, xyzn_type='xyznc', feat_type='xyzn', samp_type='dfps')
        self.pointnet2 = PointNet(planes_tab=[32, 64, 64, 64], node_radius=8, node_num=32, downsample=4, xyzn_type='xyznc', feat_type='xyzn', samp_type='dfps')
        self.pointnet3 = PointNet(planes_tab=[64, 128, 128, 128], node_radius=16, node_num=48, downsample=4, xyzn_type='xyznc', feat_type='xyzn', samp_type='dfps')
        self.pointnet4 = PointNet(planes_tab=[128, 256, 256, 256], node_radius=32, node_num=64, downsample=4, xyzn_type='xyznc', feat_type='xyzn', samp_type='dfps')
        self.mlp = Mlp(planes_tab=[256, 512, 512, 512])
        
        self.fcdown = nn.Linear(512, 512, bias=False)
        self.fcfeat = nn.Linear(512, 256, bias=False)
#        self.fc = nn.Linear(256, num_classes, bias=False)
        self.fc = angle_linear.AngleLinear(256, num_classes)
        
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, **kwargs):
        
        # otype: class/feat/image
        otype = kwargs.get('otype', 'class')
        label = kwargs.get('label', None)
               
        x = self.pointnet1(x)
        x = self.pointnet2(x)
        x = self.pointnet3(x)
        x = self.pointnet4(x)
        
        if otype == 'class' or otype == 'feat':
            x = self.mlp(x[:, 7:, ...])
            x = x.reshape(x.size(0), -1)
            
            x = self.fcdown(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.drop1(x)
            x = self.fcfeat(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.drop2(x)
        
        if otype == 'class':
            x = self.fc(x, label)

        return x
