import os
import glob
import random

import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset


class Supervised(Dataset):
    def __init__(self, root, transform=transforms.ToTensor(), ex=["png", "jpg", "jpeg"], shuffle=True):
        self.transform = transform
        self.main = []
        self.imgs = []
        self.paths = []
        self.dirs = glob.glob(f"{root}/*/")
        self.labels_map = {}
        for key, dir in enumerate(self.dirs):
            self.labels_map[key] = os.path.basename(dir[:-1])
            dirpaths = []
            for e in ex:
                dirpaths.extend(glob.glob(f"{dir}/*.{e}"))
            self.paths += dirpaths
            for path in dirpaths:
                self.imgs.append(self.transform(Image.open(path)))
            self.main.append((self.imgs[-1], key))
        if shuffle: self.shuffle()
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.main[i]
    
    def shuffle(self):
        comb = list(zip(self.paths, self.imgs, self.main))
        random.shuffle(comb)
        self.paths[:], self.imgs[:], self.main[:] = zip(*comb)

    def visualize(self, num=32, title="Images"):
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(title)
        plt.imshow(np.transpose(vutils.make_grid(self.imgs[:num],padding=True,normalize=True),(1,2,0)))
        plt.show()


class Unsupervised(Dataset):
    def __init__(self, root, transform=transforms.ToTensor(), ex=["png","jpg","jpeg"], shuffle=True):
        self.transform = transform
        self.paths = []
        for e in ex:
            self.paths.extend(glob.glob(f"{root}/*.{e}"))
        for path in self.paths:
            self.imgs.append(self.transform(Image.open(path)))
        if shuffle: self.shuffle()
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        return self.imgs[i]

    def shuffle(self):
        comb = list(zip(self.paths, self.imgs))
        random.shuffle(comb)
        self.paths[:], self.imgs[:] = zip(*comb)
    
    def visualize(self, num=32, title="Images"):
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(title)
        plt.imshow(np.transpose(vutils.make_grid(self.imgs[:num],padding=True,normalize=True),(1,2,0)))
        plt.show()
