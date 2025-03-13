import os
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
import cv2
from glob import glob

from monai.transforms import *
from monai import transforms as mt
import random
import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms as tt
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC
    
    
def normalize_images(t1):
    # Z-score normalization
    mean = t1.mean()
    std = t1.std()
    t1 = (t1 - mean) / max(std, 1e-6)
    return t1
    
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice

def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    from scipy.ndimage import binary_fill_holes
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)   # nonzero인 부분 bbox

    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]

    if seg is not None:
        seg = seg[:, :, bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


    return data, seg, bbox    


class SpineDataset(Dataset):

    def __init__(self, transform1=None, transform2=None, permutations=None):
        super(SpineDataset, self).__init__()
        
        self.transform1 = transform1
        self.transform2 = transform2
        self.permutations = permutations
        
        file_list = sorted(glob('/mnt/LSH/rin2d/spine/*.png'))
        file_list = [f for f in file_list if f.startswith('/mnt/LSH/rin2d/spine/Case')]
        file_list = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal_png/MR/*.png'))

        self.img_dir = sorted(file_list)
        
    def __len__(self):
        
        return len(self.img_dir)
    
    def __getitem__(self, idx): 
        
        label = random.randint(0,999)
        seq = torch.tensor(self.permutations[label])
        
        img = cv2.imread(self.img_dir[idx]).astype(np.float32)
        img = img[:,:,0]
    
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        
        img, _, _=crop_to_nonzero(data=img)
        
        img = np.squeeze(img, axis=0)
        # img = np.squeeze(img, axis=0)
        
        # img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)  # INTER_CUBIC
        # img = np.expand_dims(img, axis=0)
        
        img = torch.tensor(img)
        
        img = self.transform1(img)
        
        # data_dict = {'img': img}
        
        img = self.transform2(img)

        img = normalize_images(img)
        
        # gap = 16
        
        # imgclips = []
        # ori_imgclips = []
        # for i in range(3):
        #     for j in range(3):
        #         start_x = i * (64 + gap)
        #         start_y = j * (64 + gap)
                
        #         img_clip1 = img[:, i * round(64+gap/2): (i + 1) * round(64+gap/2), j * round(64+gap/2): (j + 1) * round(64+gap/2)]   # jigsaw용 img
        #         ori_imgclips.append(img_clip1)             
        #         # img_clip2 = img[:, i * 64: (i + 1) * 64, j * 64: (j + 1 + self.args.gap) * 64]
        #         piece = img[:, start_x:start_x+64, start_y:start_y+64]
        #         imgclips.append(piece)
        
        # imgclips = [imgclips[item] for item in self.permutations[label]]
        # imgclips = torch.stack(imgclips, dim=0)
        
        
        data_dict={'img':img, 'seq':seq, 'label':label}    
        
        
        return data_dict
        # return img, seq, torch.tensor(label).long()
    
    

# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
#     return img


# class ImageNetDataset(DatasetFolder):
#     def __init__(
#             self,
#             imagenet_folder: str,
#             train: bool,
#             transform: Callable,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#     ):
#         imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
#         super(ImageNetDataset, self).__init__(
#             imagenet_folder,
#             loader=pil_loader,
#             extensions=IMG_EXTENSIONS if is_valid_file is None else None,
#             transform=transform,
#             target_transform=None, is_valid_file=is_valid_file
#         )
        
#         self.samples = tuple(img for (img, label) in self.samples)
#         self.targets = None # this is self-supervised learning so we don't need labels
    
#     def __getitem__(self, index: int) -> Any:
#         img_file_path = self.samples[index]
#         return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(input_size, permutations=None) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    trans_train1 = tt.Compose([
        tt.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        # tt.RandomHorizontalFlip()
    ])
    trans_train2 = mt.Compose([ mt.RandGaussianNoise(mean=0, std=0.1, prob=0.2),
                                mt.RandGaussianSmooth(sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.2),
                                mt.RandScaleIntensity(factors=0.25, prob=0.2),
                                mt.RandAdjustContrast(gamma=(0.75, 1.25), prob=0.2),
                                # mt.ToTensord(keys=["img"])
                                ])
    
    # dataset_path = os.path.abspath(dataset_path)
    # for postfix in ('train', 'val'):
    #     if dataset_path.endswith(postfix):
    #         dataset_path = dataset_path[:-len(postfix)]
    
    dataset_train = SpineDataset(transform1=trans_train1, transform2=trans_train2, permutations=permutations)
    print_transform(trans_train1, '[pre-train]')
    print_transform(trans_train2, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
