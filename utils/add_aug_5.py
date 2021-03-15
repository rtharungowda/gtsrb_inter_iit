import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import os
import numpy as np
import glob

# transform1 = A.Compose([
#                 A.Rotate(always_apply=False, p=1.0, limit=(-24, 24), interpolation=0,
#                         border_mode=0, value=(0, 0, 0), mask_value=None),

#                 A.OneOf([
#                     A.GaussNoise(var_limit=(10.0, 210.52999877929688)),
#                     A.ISONoise(intensity=(0.10000000149011612, 1.5),
#                             color_shift=(0.03999999910593033, 0.4099999964237213)),
#                 ],p=0.7),
                
#                 A.OneOf([
#                     A.RandomRain(slant_lower=-9, slant_upper=9, 
#                                 drop_length=24, drop_width=1, drop_color=(0, 0, 0), blur_value=5, 
#                                 brightness_coefficient=0.6299999952316284, rain_type=None),

#                     A.RandomFog(fog_coef_lower=0.10000000149011612,
#                                 fog_coef_upper=0.5399999618530273, alpha_coef=0.7799999713897705),

#                     A.RandomSnow(snow_point_lower=0.10000000149011612,
#                                 snow_point_upper=0.28999999165534973, brightness_coeff=1.5299999713897705),
#                 ],p=1.0),

#                 A.OpticalDistortion(p=0.4, distort_limit=(-0.6399999856948853, 0.6399999856948853),
#                             shift_limit=(-0.20999999344348907, 0.20999999344348907), interpolation=0, border_mode=2, 
#                             value=(0, 0, 0), mask_value=None),
#         ])
t1 = A.OneOf([
        A.RandomRain(slant_lower=-9, slant_upper=9, 
                    drop_length=24, drop_width=1, drop_color=(0, 0, 0), blur_value=5, 
                    brightness_coefficient=0.6299999952316284, rain_type=None),

        A.RandomFog(fog_coef_lower=0.10000000149011612,
                    fog_coef_upper=0.5399999618530273, alpha_coef=0.7799999713897705),

        A.RandomSnow(snow_point_lower=0.10000000149011612,
                    snow_point_upper=0.28999999165534973, brightness_coeff=1.5299999713897705),
],p=1.0)

t2 = A.Compose([
    A.Rotate(always_apply=False, p=0.5, limit=(-24, 24), interpolation=0,
                        border_mode=0, value=(0, 0, 0), mask_value=None),

    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 210.52999877929688)),
        A.ISONoise(intensity=(0.10000000149011612, 1.5),
                color_shift=(0.03999999910593033, 0.4099999964237213)),
    ],p=0.5),
])

t3 = A.OneOf([
    A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-1.2300000190734863, 0.12999999523162842),
                    shift_limit=(-0.5, 0.5099999904632568), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
    A.GridDistortion(always_apply=False, p=1.0, num_steps=4, distort_limit=(-0.29999998211860657, 0.30000001192092896),
                    interpolation=0, border_mode=2, value=(0, 0, 0), mask_value=None)
],p=1.0)

def make_img(path,folder):
    img = Image.open(path)
    img = np.array(img)
    base_ = os.path.basename(path)
    filename = base_.split()[0]
    print(filename)

    fld = os.path.join('/content/drive/MyDrive/Bosch/Aug_new_5_classes',folder)
    flname = os.path.join(fld,filename)
    if os.path.isdir(fld) == False:
        os.mkdir(fld)

    x1 = t1(image=img)['image']
    nx1 = Image.fromarray(x1)
    nx1.save(flname+'_01.png')

    x2 = t2(image=img)['image']
    nx2 = Image.fromarray(x2)
    nx2.save(flname+'_02.png')

    x3 = t3(image=img)['image']
    nx3 = Image.fromarray(x3)
    nx3.save(flname+'_03.png')

if __name__ == "__main__":
    for i in range(43,48):
        print(f'folder {i}')
        path = '/content/drive/MyDrive/Bosch/New Dataset/000'+str(i)
        ext = ['jpg','jpeg','png','ppm']
        files = []
        print('loading.....')
        [files.extend(glob.glob(path + '/*.' + e)) for e in ext]
        for j in range(len(files)):
           pth = os.path.join(path,files[j])
        #    print(pth)
           make_img(pth,'000'+str(i))