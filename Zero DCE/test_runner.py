import torch
import os
import numpy as np
from PIL import Image
import model
import torchvision
import glob

def low_light(image_path):
    image = Image.open(image_path)
    image = np.asarray(image) / 255.0   
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)

    DCE_Net = model.DCENet()
    DCE_Net.load_state_dict(torch.load('snapshots_0.5/Epoch99.pth', map_location=torch.device('cpu')))
    _, result, _ = DCE_Net(image)

    res_path = "results_99_0.5/" + image_path.split('/')[-1]

    if not os.path.exists('results_99_0.5'):
        os.makedirs('results_99_0.5')

    torchvision.utils.save_image(result, res_path)

if __name__ == '__main__':
    with torch.no_grad():
        test_path = '/Applications/ML projects/Success/Low Light Image Enhancement/lol_dataset/eval15/low/'
        test_list = glob.glob(test_path + '*.png')

        for image in test_list:
            low_light(image)
