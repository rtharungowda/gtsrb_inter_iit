from PIL import Image
import glob
import os
import pandas as pd
import numpy as np
import torch

def save_ckp(state, checkpoint_path):
    f_path = checkpoint_path
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer, device):

    checkpoint = torch.load(checkpoint_fpath,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_acc = checkpoint['valid_acc'] 
    return model, optimizer, checkpoint['epoch'], valid_acc

def calc_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    for data, _ in loader:
        channels_sum +=torch.mean(data,dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches+=1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    
    return mean,std
        

def save_in_jpg(path):
    im = Image.open(path)
    print(im)
    base_path = os.path.basename(path)
    im.save(os.path.splitext(base_path)[0]+".jpg")

def create_csv(path1,path2):
    folders1 = os.listdir(path1)
    for i in range(len(folders1)):
        folders1[i]=os.path.join(path1,folders1[i])
    folders2 = os.listdir(path2)
    for i in range(len(folders2)):
        folders2[i]=os.path.join(path2,folders2[i])
    folders = []
    folders.extend(folders1)
    folders.extend(folders2)
    label = []
    img_pth = []
    for folder in folders:
        folder_path = folder
        
        # print(folder_path)
        ext = ['jpg','jpeg','png','ppm']
        # files = glob.glob(folder_path+"/*.png")
        files = []
        print('loading.....')
        [files.extend(glob.glob(folder_path + '/*.' + e)) for e in ext]
        print(f"folder-{folder}, images-{len(files)}")
        for file in files:
            label.append(int(folder.split('/')[-1]))
            file_path = os.path.join(folder_path,file)
            img_pth.append(file_path)
            # print(file_path)
    
    df = pd.DataFrame(list(zip(img_pth,label)),columns=["path","label"])
    classes = pd.unique(df['label'])
    print(f"number of classes {len(classes)}, classes {classes} and number of examples {len(df['path'])}")
    df.to_csv("/content/gtsrb_inter_iit/utils/gtsrb_train_all_5aug.csv")

def plots(path):
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)
    print(df.groupby('label').count())

if __name__ == "__main__":
    # save_in_jpg("/content/drive/MyDrive/Bosch/imgs/00019/00000_00001.ppm")
    # create_csv("/content/drive/MyDrive/Bosch/imgs","/content/drive/MyDrive/Bosch/Aug_new_5_classes")
    plots("/content/gtsrb_inter_iit/utils/gtsrb_train_all_5aug.csv")