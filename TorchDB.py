import numpy as np
import os
from PIL import Image
import glob
import random
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms


class DBreader_frame_interpolation(Dataset):
    """
    DBreader reads all triplet set of frames in a directory.
    Each triplet set contains frame 0, 1, 2.
    Each image is named frame0.png, frame1.png, frame2.png.
    Frame 0, 2 are the input and frame 1 is the output.
    """

    def __init__(self, db_dir, resize=None):
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.db_dir = db_dir
        self.list_classes = os.listdir(db_dir)

    def make_basename(self, className, index):
        index = str(index+1)
        try:
            return os.path.join(self.db_dir, className, '0'*(4-len(index)) + index + '.jpg')
        except:
            return os.path.join(self.db_dir, className, '0'*(4-len(index)) + index + '.png')

    def __getitem__(self, index):
        curr_class = self.list_classes[index % len(self.list_classes)]
        curr_class_path = os.path.join(self.db_dir, curr_class)
        index = max(0, index % (len(os.listdir(curr_class_path))) - 2)
        frame0 = self.transform(Image.open(self.make_basename(curr_class, index)))
        frame1 = self.transform(Image.open(self.make_basename(curr_class, index + 1)))
        frame2 = self.transform(Image.open(self.make_basename(curr_class, index + 2)))

        return frame0, frame1, frame2

    def __len__(self):
        # return self.file_len
        total = 0
        for class_ in self.list_classes:
            total += len(glob.glob(self.db_dir + '/' + class_ + '/*.jpg') + glob.glob(self.db_dir + '/' + class_ + '/*.png')) - 2
        
        return total
