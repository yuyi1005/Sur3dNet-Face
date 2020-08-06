# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:17:17 2019

@author: Administrator
"""

import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def show_scatter3d(x, y, z, c=None):

    fig = plt.figure()
    for i in range(c.size(0)):
        sqsz = round(math.sqrt(c.size(0)))
        ax = fig.add_subplot(sqsz, (c.size(0) + sqsz - 1) / sqsz, i + 1, projection='3d')
        ax.axis('equal')
        ax.axis('off')
        ax.view_init(90, -90)
        ax.scatter(x, y, z, s=1, c=c[i, :], marker='.')
        
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.show()

def show_tripcolor(x, y, z, c=None):

    fig = plt.figure()
    for i in range(c.size(0)):
        sqsz = round(math.sqrt(c.size(0)))
        ax = fig.add_subplot(sqsz, (c.size(0) + sqsz - 1) / sqsz, i + 1)
        ax.axis('equal')
        ax.axis('off')
        ax.tripcolor(x, y, c[i, :], shading='gouraud')
        
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.show()

def show_scatter(x, y, z, c=None):

    fig = plt.figure()
    for i in range(c.size(0)):
        sqsz = round(math.sqrt(c.size(0)))
        ax = fig.add_subplot(sqsz, (c.size(0) + sqsz - 1) / sqsz, i + 1)
        ax.axis('equal')
        ax.axis('off')
        ax.scatter(x, y, 0.1, c[i, :])
        
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.show()
