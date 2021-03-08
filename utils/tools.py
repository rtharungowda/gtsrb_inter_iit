from PIL import Image
import glob
import os
import pandas as pd
import numpy as np
import torch

def calc_mean_std(loader):
    # imgs = []
    # for ind,row in df.iterrows():
    #     print(row['path'])
    #     img = Image.open(row['path'])
    #     img = img.resize((30,40))
    #     img_ar = np.array(img)
    #     imgs.append(img_ar)
    # ar = np.array(imgs)
    # mean, std = ar.mean(axis=0), ar.std(axis=0)
    # print(mean,std)
    # return mean ,std

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

def create_csv(path):
    folders = os.listdir(path)
    label = []
    img_pth = []
    for folder in folders:
        folder_path = os.path.join(path,folder)
        # print(folder_path)
        files = glob.glob(folder_path+"/*.ppm")
        print(f"folder-{folder}, images-{len(files)}")
        for file in files:
            label.append(int(folder))
            file_path = os.path.join(folder_path,file)
            img_pth.append(file_path)
            # print(file_path)
    
    df = pd.DataFrame(list(zip(img_pth,label)),columns=["path","label"])
    classes = pd.unique(df['label'])
    print(f"number of classes {classes} and number of examples {len(df['path'])}")
    df.to_csv("gtsrb_train.csv")

def plots(path):
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)
    print(df.groupby('label').count())

if __name__ == "__main__":
    # save_in_jpg("/content/drive/MyDrive/Bosch/imgs/00019/00000_00001.ppm")
    # create_csv("/content/drive/MyDrive/Bosch/imgs")
    plots("/content/gtsrb_inter_iit/utils/gtsrb_train.csv")