import os

import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

import sys
sys.path.insert(1,'/content/gtsrb_inter_iit/engine/')
sys.path.insert(1,'/content/gtsrb_inter_iit/utils/')
from model import TrafficSignNet
from dataloader import preprocess
from tools import load_ckp

DEVICE = torch.device("cpu")

def get_classes():
    return [str(i) for i in range(0,43)]

def baseline_func(input):
    return input * 0

def formatted_data_iter():
    
    path = "/content/gtsrb_inter_iit/utils/gtsrb_test.csv"
    print(path)
    df = pd.read_csv(path)
    dataset_size ,dataloader = preprocess(df,test=True,batch_size=64)

    loader = iter(dataloader)

    while True:
        images, labels = next(loader)
        yield Batch(inputs=images, labels=labels)

if __name__=="__main__":
    model = TrafficSignNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    model, _, _, _ = load_ckp("/content/drive/MyDrive/competitions/bosh-inter-iit/model3.pt", model, optimizer, DEVICE)
    norm = transforms.Normalize([0.3401, 0.3120, 0.3212],[0.2725, 0.2609, 0.2669])
    visualizer = AttributionVisualizer(
        models=[model],
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=get_classes(),
        features=[
            ImageFeature(
                "Photo",
                baseline_transforms=[baseline_func],
                input_transforms=[norm],
            )
        ],
        dataset=formatted_data_iter(),
    )

    visualizer.render()