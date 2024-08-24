import numpy as np
import glob
import random
from PIL import Image
import torch

def populate_list(path):
    list = glob.glob(path + "*.png")
    random.shuffle(list)

    return list

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, path, image_size=256):
        self.data_list = populate_list(path)
        self.image_size = image_size

    def __getitem__(self, index):
        data_path = self.data_list[index]
        image = Image.open(data_path)
        image = image.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        image = np.asarray(image) / 255.0
        image = torch.from_numpy(image).float()

        return image.permute(2, 0, 1)
    
    def __len__(self):
        return len(self.data_list)