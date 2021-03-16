from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np
# import Augmentor 

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(1,'/content/gtsrb_inter_iit/utils')

from tools import calc_mean_std

class GTSRB(Dataset):
    def __init__(self,
                dataframe,
                transforms):
        self.df = dataframe
        self.tf = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        x = Image.open(self.df.iloc[index]['path'])
        y = self.df.iloc[index]['label']

        x = np.array(x)
        x = self.tf(image=x)['image']

        return x,y
    
def mean_std(df, ratio=0.2):
    df_train, _ = train_test_split(df,
                                    stratify=df['label'],
                                    test_size=ratio,
                                    random_state=42)
    train_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    train_dataset = GTSRB(dataframe=df_train, transforms=train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    mean, std = calc_mean_std(train_loader)
    return mean,std 

def preprocess(gtsrb, test=False ,ratio=0.2, batch_size = 64):

    if test:
        # test_transforms = A.Compose([
        #     A.Resize(width=32, height=32),
        #     A.Normalize(mean=[0.3401, 0.3120, 0.3212], std=[0.2725, 0.2609, 0.2669]),
        #     ToTensorV2(),
        # ])

        test_transforms = A.Compose([
            A.Resize(width=32, height=32),
            # A.Rotate(always_apply=False, p=1.0, limit=(-24, 24), interpolation=0,
            #          border_mode=0, value=(0, 0, 0), mask_value=None),

            # A.OneOf([
            #     A.GaussNoise(var_limit=(10.0, 210.52999877929688)),
            #     A.ISONoise(intensity=(0.10000000149011612, 1.5),
            #                color_shift=(0.03999999910593033, 0.4099999964237213)),
            # ],p=0.7),
            
            # A.OneOf([
            #     A.RandomRain(slant_lower=-9, slant_upper=9, 
            #                 drop_length=24, drop_width=1, drop_color=(0, 0, 0), blur_value=5, 
            #                 brightness_coefficient=0.6299999952316284, rain_type=None),

            #     A.RandomFog(fog_coef_lower=0.10000000149011612,
            #                 fog_coef_upper=0.5399999618530273, alpha_coef=0.7799999713897705),

            #     A.RandomSnow(snow_point_lower=0.10000000149011612,
            #                 snow_point_upper=0.28999999165534973, brightness_coeff=1.5299999713897705),
            # ],p=0.8),

            # A.OpticalDistortion(p=0.4, distort_limit=(-0.6399999856948853, 0.6399999856948853),
            #              shift_limit=(-0.20999999344348907, 0.20999999344348907), interpolation=0, border_mode=2, 
            #              value=(0, 0, 0), mask_value=None),

            A.Normalize(mean=[0.3401, 0.3120, 0.3212], std=[0.2725, 0.2609, 0.2669]),
            ToTensorV2(),
        ])

        test_dataset = GTSRB(dataframe=gtsrb, transforms=test_transforms)

        test_dataloader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        return len(test_dataset), test_dataloader

    num_classes = gtsrb['label'].nunique()

    df_train, df_val = train_test_split(gtsrb,
                                        stratify=gtsrb['label'],
                                        test_size=ratio,
                                        random_state=42)

    df_train.to_csv('/content/gtsrb_inter_iit/utils/gtsrb_all_split_train.csv')
    df_val.to_csv('/content/gtsrb_inter_iit/utils/gtsrb_all_split_val.csv')

    print(f"train imgs = {len(df_train)} val imgs = {len(df_val)}")
    df = {'train':df_train,'val':df_val}

    # mean (tensor([0.3401, 0.3120, 0.3212]), std tensor([0.2725, 0.2609, 0.2669]))

    data_transforms = {
        "train": A.Compose([
            A.Resize(width=32, height=32),
            # A.Rotate(always_apply=False, p=1.0, limit=(-24, 24), interpolation=0,
            #          border_mode=0, value=(0, 0, 0), mask_value=None),

            # A.OneOf([
            #     A.GaussNoise(var_limit=(10.0, 210.52999877929688)),
            #     A.ISONoise(intensity=(0.10000000149011612, 1.5),
            #                color_shift=(0.03999999910593033, 0.4099999964237213)),
            # ],p=0.7),
            
            # A.OneOf([
            #     A.RandomRain(slant_lower=-9, slant_upper=9, 
            #                 drop_length=24, drop_width=1, drop_color=(0, 0, 0), blur_value=5, 
            #                 brightness_coefficient=0.6299999952316284, rain_type=None),

            #     A.RandomFog(fog_coef_lower=0.10000000149011612,
            #                 fog_coef_upper=0.5399999618530273, alpha_coef=0.7799999713897705),

            #     A.RandomSnow(snow_point_lower=0.10000000149011612,
            #                 snow_point_upper=0.28999999165534973, brightness_coeff=1.5299999713897705),
            # ],p=0.8),

            # A.OpticalDistortion(p=0.4, distort_limit=(-0.6399999856948853, 0.6399999856948853),
            #              shift_limit=(-0.20999999344348907, 0.20999999344348907), interpolation=0, border_mode=2, 
            #              value=(0, 0, 0), mask_value=None),

            A.Normalize(mean=[0.3401, 0.3120, 0.3212], std=[0.2725, 0.2609, 0.2669]),
            ToTensorV2(),
        ]),
        "val":A.Compose([
            A.Resize(width=32, height=32),
            # A.Rotate(always_apply=False, p=1.0, limit=(-24, 24), interpolation=0,
            #          border_mode=0, value=(0, 0, 0), mask_value=None),

            # A.OneOf([
            #     A.GaussNoise(var_limit=(10.0, 210.52999877929688)),
            #     A.ISONoise(intensity=(0.10000000149011612, 1.5),
            #                color_shift=(0.03999999910593033, 0.4099999964237213)),
            # ],p=0.7),
            
            # A.OneOf([
            #     A.RandomRain(slant_lower=-9, slant_upper=9, 
            #                 drop_length=24, drop_width=1, drop_color=(0, 0, 0), blur_value=5, 
            #                 brightness_coefficient=0.6299999952316284, rain_type=None),

            #     A.RandomFog(fog_coef_lower=0.10000000149011612,
            #                 fog_coef_upper=0.5399999618530273, alpha_coef=0.7799999713897705),

            #     A.RandomSnow(snow_point_lower=0.10000000149011612,
            #                 snow_point_upper=0.28999999165534973, brightness_coeff=1.5299999713897705),
            # ],p=0.8),

            # A.OpticalDistortion(p=0.4, distort_limit=(-0.6399999856948853, 0.6399999856948853),
            #              shift_limit=(-0.20999999344348907, 0.20999999344348907), interpolation=0, border_mode=2, 
            #              value=(0, 0, 0), mask_value=None),

            A.Normalize(mean=[0.3401, 0.3120, 0.3212], std=[0.2725, 0.2609, 0.2669]),
            ToTensorV2(),
        ])
    }


    img_dataset = {x: GTSRB(dataframe=df[x], transforms=data_transforms[x]) 
                for x in ['train','val']
            }

    dataset_sizes = {x: len(img_dataset[x]) for x in ['train','val']}

    dataloader = {x: DataLoader(img_dataset[x], batch_size=batch_size, shuffle=True, num_workers=2)
                for x in ['train','val']
            }

    return dataset_sizes, dataloader