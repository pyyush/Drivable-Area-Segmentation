#Reference: ucbdrive https://github.com/ucbdrive/bdd-data
#Modified by Piyush Vyas For LaneNet: Drivable Area Segmentation

import os
from os import path
import sys


def gen_list(data_root, data_dir, list_dir, phase, list_type, suffix='.jpg'):
    phase_dir = path.join(data_root, data_dir, phase)
    images = sorted([path.join(data_dir, phase, n)
                     for n in os.listdir(phase_dir)
                     if n[-len(suffix):] == suffix])
    print('Found', len(images), 'items in', data_dir, phase)
    out_path = path.join(list_dir, '{}_{}.txt'.format(phase, list_type))
    if not path.exists(list_dir):
        os.makedirs(list_dir)
    print('Writing', out_path)
    with open(out_path, 'w') as fp:
        fp.write('\n'.join(images))

def gen_images(data_root, list_dir):
    for phase in ['train', 'val', 'test']:
        gen_list(data_root, path.join('images', '100k'),
                 list_dir, phase, 'images', '.jpg')

def gen_drivable(data_root):
    label_dir = 'drivable_maps/labels'
    list_dir = 'lists'
    gen_images(data_root, list_dir)
    for p in ['train', 'val']:
        gen_list(data_root, label_dir, list_dir, p, 'labels', 'drivable_id.png') #'drivable_color.png' for color labels


gen_drivable('bdd100k/') #sys.argv[])
