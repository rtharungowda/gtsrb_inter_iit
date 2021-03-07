from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

import numpy as np

def calc_mean_std(df):
    imgs = []
    for ind,row in df.iterrows():
        print(row['path'])
        img = Image.open(row['path'])
        img = img.resize((30,40))
        img_ar = np.array(img)
        imgs.append(img_ar)
        
    ar = np.array(imgs)
    mean, std = ar.mean(axis=0), ar.std(axis=0)
    print(mean,std)
    return mean ,std

class GTSRB(Dataset):
    def __init__(self,
                dataframe,
                transforms = None):
        self.dataframe = df
        self.tf = transforms
    
    def __len__(self):
        return len(df)
    
    def __getitem__(self,index):
        x = Image.open(self.df['path'].iloc[index])
        y = self.df['label'].iloc[index]

        if self.tf is not None:
            x = self.tf(x)

        return x,y

def preprocess(gtsrb, ratio=0.2):

    num_classes = gtsrb['label'].nunique()

    df_train, df_val = train_test_split(gtsrb,
                                        stratify=gtsrb['label'],
                                        test_size=ratio)
    print(f"train imgs = {len(df_train)} val imgs = {len(df_val)}")
    df = {'train':df_train,'val':df_val}

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((30,40)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((30,40)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])
    }


    dataset = {x: GTSRB(dataframe=df[x], transforms=data_transforms[x]) 
                for x in ['train','val']
            }

    dataloader = {x: DataLoader(dataset[x], batch_size=64, shuffle=True, num_workers=4)
                for x in ['train','val']
            }



if __name__ == "__main__":
    df = pd.read_csv("/content/gtsrb_inter_iit/utils/gtsrb_train.csv")
    # preprocess(df,0.1)
    calc_mean_std(df)