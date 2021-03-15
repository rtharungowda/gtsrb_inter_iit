import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import  DataLoader
from torchvision import models

import torchvision.transforms as transforms
import torchvision.datasets as dataset

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import sys
sys.path.insert(1,'/content/gtsrb_inter_iit/engine/')
sys.path.insert(1,'/content/gtsrb_inter_iit/utils/')
from model import TrafficSignNet
from dataloader import preprocess
from tools import load_ckp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_output(path):
    img = Image.open(path)

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3401, 0.3120, 0.3212], std=[0.2725, 0.2609, 0.2669]),
    ])

    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.size())

    no_of_layers=0
    conv_layers=[]

    model_children=list(model.children())

    for child in model_children :
        if type(child)==nn.Conv2d :
            no_of_layers+=1
            conv_layers.append(child)
        elif type(child)==nn.Sequential :
            for layer in child.children():
                if type(layer)==nn.Conv2d :
                    no_of_layers+=1
                    conv_layers.append(layer)
    print(conv_layers)
    print(no_of_layers)

    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    outputs = results

    for num_layer in range(len(outputs)):
        plt.figure(figsize=(50, 100))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print("Layer ",num_layer+1)
        num_filt = len(layer_viz)
        print(num_filt)
        for i, filter in enumerate(layer_viz):
            # if i == 16: 
            #     break
            plt.subplot(num_filt/5, 5, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        plt.savefig(f'/content/gtsrb_inter_iit/utils/layer{str(num_layer)}.png')
        plt.clf()

if __name__ == '__main__':
    model = TrafficSignNet(num_classes=48)
    optimizer = optim.Adam(model.parameters())
    model = model.to(DEVICE)
    model, _, _, _ = load_ckp("/content/drive/MyDrive/competitions/bosh-inter-iit/48_classes_album2.pt", model, optimizer, DEVICE)
    # print(model)
    conv_output(path='/content/drive/MyDrive/Bosch/New Dataset/00020/00000_00004.jpg')
    