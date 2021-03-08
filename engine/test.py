import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim

import pandas as pd
import os

import sys
sys.path.insert(1,'/content/gtsrb_inter_iit/engine/')
sys.path.insert(1,'/content/gtsrb_inter_iit/utils/')
from model import TrafficSignNet
from dataloader import preprocess
from tools import load_ckp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prep_csv(path, image_folder):
    df = pd.read_csv(path, sep=';')

    def append(x):
        return os.path.join(image_folder,x)
    df['Filename'] = df['Filename'].apply(append)

    df = df.rename(columns={'Filename':'path','ClassId':'label'})
    print(df.head)

    data_columns=['path','label']
    df.loc[:,data_columns].to_csv("gtsrb_test.csv")

if __name__ == "__main__":
    prep_csv("/content/GTSRB/GT-final_test.csv","/content/GTSRB/Final_Test/Images")
    df = pd.read_csv("/content/gtsrb_inter_iit/utils/gtsrb_test.csv")
    dataset_size ,dataloader = preprocess(df,test=True)
    model = TrafficSignNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    model, optimizer, _, _ = load_ckp("/content/drive/MyDrive/competitions/bosh-inter-iit/model2.pt", model, optimizer)