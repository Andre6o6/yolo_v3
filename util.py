from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    #prediction - output tensor B x C x H x W 
    #inp_dim - input image dimension

    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes    #{bx, by, bw, bh, objective score} + classes probabilities
    num_anchors = len(anchors)
    
    #Reshape prediction to B x (# of bbox attributes in one cell) x (# of grid cells)
    #!ERROR HERE
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    #Swap last 2 dimentions. Call .contiguous() so that it will affect not only indexes, but data itself
    prediction = prediction.transpose(1,2).contiguous()
    #Reshape prediction to B x (# of bboxes) x (# of bbox attributes in one anchor)
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    #The dimensions of the anchors are in accordance to the height and width attributes of the net block.
    #These attributes describe the dimensions of the input image, 
    #which is larger (by a factor of stride) than the detection map. 
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset  #iiuc add grid cell's corner coordinates to bbox offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Apply sigmoid activation to the the class scores
    prediction[:,:,5:5 + num_classes] = torch.sigmoid((prediction[:,:, 5:5 + num_classes]))

    #Resize the detections map to the size of the input image.
    #The bounding box attributes here are sized according to the feature map. 
    #We multiply the attributes by the stride variable.
    prediction[:,:,:4] *= stride

    return prediction

