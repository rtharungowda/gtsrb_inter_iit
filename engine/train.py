import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import pandas as pd
from PIL import Image

import numpy as np
import time
import copy
import matplotlib.pyplot as plt

import sys
sys.path.insert(1,'/content/gtsrb_inter_iit/engine/')
sys.path.insert(1,'/content/gtsrb_inter_iit/utils/')
from model import TrafficSignNet
from dataloader import preprocess
from tools import save_ckp, load_ckp

from torch.utils.tensorboard import SummaryWriter

EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LR = 0.001
BATCH_SIZE = 64

scaler = torch.cuda.amp.GradScaler()

def train_model(model, 
                criterion, 
                optimizer, 
                dataloaders,
                dataset_sizes, 
                scheduler=None):

    device = DEVICE
    num_epochs = EPOCHS
    model = model.to(device)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # writer = SummaryWriter(
    #         f"runs/bs_{BATCH_SIZE}_LR_{LR}"
    #     )

    # images, _ = next(iter(dataloaders['train']))
    # writer.add_graph(model, images.to(device))
    # writer.close()

    loss_p = {'train':[],'val':[]}
    acc_p = {'train':[],'val':[]}
    f1_p = {'train':[],'val':[]}
    not_imp = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # loss.backward()
                        # optimizer.step()

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                preds_on = preds.to('cpu').tolist()
                labels_on = labels.data.to('cpu').tolist()
                all_preds.extend(preds_on)
                all_labels.extend(labels_on)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                print(f"running_loss {running_loss} running_corrects {running_corrects}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            F1_score = f1_score( all_labels, all_preds, zero_division=1, average='weighted')

            loss_p[phase].append(epoch_loss)
            acc_p[phase].append(epoch_acc)
            f1_p[phase].append(F1_score)

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                not_imp = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'valid_acc': best_acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                #save checkpoint
                checkpoint_path = "/content/drive/MyDrive/competitions/bosh-inter-iit/48_classes_album3_aug.pt"
                save_ckp(checkpoint, checkpoint_path)
            elif phase=='val' and epoch_acc < best_acc:
                not_imp += 1

            if not_imp > 15:
                break 

        if not_imp > 15:
            break             
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, loss_p, acc_p, f1_p


if __name__ == "__main__":
    df = pd.read_csv("/content/gtsrb_inter_iit/utils/gtsrb_train_all_5aug.csv")
    dataset_sizes,dataloaders = preprocess(df,ratio=0.1,batch_size=BATCH_SIZE)
    num_classes = df['label'].nunique()
    print(f"number of classes {num_classes}")
    model = TrafficSignNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # model = model.to(DEVICE)
    # model, _, _, _ = load_ckp("/content/drive/MyDrive/competitions/bosh-inter-iit/48_classes_album2.pt", model, optimizer, DEVICE)

    final_model, best_acc, loss_p, acc_p, f1_p = train_model(model,criterion,optimizer,dataloaders,dataset_sizes)

    print("loss dict",loss_p)
    print("train dict",acc_p)
    print("f1 dict",f1_p)
    # checkpoint = {
    #         'epoch': EPOCHS,
    #         'valid_acc': best_acc,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }
    

    # loss_p = {'train':[10,20,30,45],'val':[45,20,30,45]}
    # acc_p = {'train':[100,20,30,45],'val':[10,0,30,45]}
    # f1_p = {'train':[10,2,30,5],'val':[1,20,3,45]}
    # EPOCHS = 4
    x = [i for i in range(EPOCHS)]
    print(x)

    #loss
    plt.plot(x,loss_p['train'],color='red', marker='o')
    plt.title('Train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/gtsrb_inter_iit/utils/train_loss.png')
    plt.clf()

    plt.plot(x, loss_p['val'],color='red', marker='o')
    plt.title('Val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True) 
    plt.savefig('/content/gtsrb_inter_iit/utils/val_loss.png')
    plt.clf()
    
    #acc
    plt.plot(x, acc_p['train'],color='red', marker='o')
    plt.title('Train acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/gtsrb_inter_iit/utils/train_acc.png')
    plt.clf()

    plt.plot(x, acc_p['val'],color='red', marker='o')
    plt.title('Val acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid(True) 
    plt.savefig('/content/gtsrb_inter_iit/utils/val_acc.png')
    plt.clf()

    #f1 
    plt.plot(x, f1_p['train'],color='red', marker='o')
    plt.title('Train f1')
    plt.xlabel('epochs')
    plt.ylabel('f1')
    plt.grid(True) 
    plt.savefig('/content/gtsrb_inter_iit/utils/train_f1.png')
    plt.clf()

    plt.plot(x, f1_p['val'],color='red', marker='o')
    plt.title('Val f1')
    plt.xlabel('epochs')
    plt.ylabel('f1')
    plt.grid(True) 
    plt.savefig('/content/gtsrb_inter_iit/utils/val_f1.png') 
    plt.clf()   
    # # save checkpoint
    # checkpoint_path = "/content/drive/MyDrive/competitions/bosh-inter-iit/model4.pt"
    # save_ckp(checkpoint, checkpoint_path)