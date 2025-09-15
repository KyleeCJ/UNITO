import os
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.image_dir = data_dir.iloc[:,0].to_list()
    # self.mask_dir = data_dir.iloc[:,1].to_list()
    self.transform = transform
  
  def __len__(self):
    return len(self.data_dir)

  def __getitem__(self, index):
    img_path = self.image_dir[index]
    # mask_path = self.mask_dir[index]
    image = np.load(img_path,allow_pickle=True).astype('double')
    image = image/image.max()
    # mask = np.load(mask_path,allow_pickle=True).astype('double')
    # mask[mask==255] = 1.0

    if self.transform is not None:
      augmentations = self.transform(image=image)
      image = augmentations["image"]
      # mask = augmentations["mask"]
      
    return image, img_path