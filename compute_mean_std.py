#Ref: original version by Fisher Yu https://github.com/fyu/drn
#Modified by Piyush Vyas for LaneNet: Drivable Area Segmentation
import json
import numpy as np
from PIL import Image
from os import path as osp


def compute_mean_std(list_dir, data_dir):
    image_list_path = osp.join(list_dir, 'train_images.txt')
    image_list = [line.strip() for line in open(image_list_path, 'r')]
    np.random.shuffle(image_list)
    pixels = []
    for image_path in image_list[:500]:
        image = Image.open(osp.join(data_dir, image_path), 'r')
        pixels.append(np.asarray(image).reshape(-1, 3))
    pixels = np.vstack(pixels)
    mean = np.mean(pixels, axis=0) / 255
    std = np.std(pixels, axis=0) / 255
    print(mean, std)
    info = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(osp.join(data_dir, 'info.json'), 'w') as fp:
        json.dump(info, fp)


compute_mean_std('bdd100k/lists/', 'bdd100k/')


