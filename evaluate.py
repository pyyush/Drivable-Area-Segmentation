#Ref: Author Berkeley DeepDrive
#Official Evaluation Script  for Drivable Area
#Modified by Piyush Vyas for LaneNet: Drivable Area Segmentation
import argparse
import json
import os
from collections import defaultdict
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', choices=['seg', 'det', 'drivable',
                        'det-tracking'])
    parser.add_argument('--gt', '-g', help='path to ground truth')
    parser.add_argument('--result', '-r',
                        help='path to results to be evaluated')
    parser.add_argument('--categories', '-c', nargs='+',
                        help='categories to keep')
    args = parser.parse_args()

    return args


def fast_hist(gt, prediction, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(
        n * gt[k].astype(int) + prediction[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious


def find_all_png(folder):
    paths = []
    for root, dirs, files in os.walk(folder, topdown=True):
        paths.extend([osp.join(root, f)
                      for f in files if osp.splitext(f)[1] == '.png'])
    return paths


def evaluate_segmentation(gt_dir, result_dir, num_classes, key_length):
    gt_dict = dict([(osp.split(p)[1][:key_length], p)
                    for p in find_all_png(gt_dir)])
    result_dict = dict([(osp.split(p)[1][:key_length], p)
                        for p in find_all_png(result_dir)])
    result_gt_keys = set(gt_dict.keys()) & set(result_dict.keys())
    if len(result_gt_keys) != len(gt_dict):
        raise ValueError('Result folder only has {} of {} ground truth files.'
                         .format(len(result_gt_keys), len(gt_dict)))
    print('Found', len(result_dict), 'results')
    print('Evaluating', len(gt_dict), 'results')
    hist = np.zeros((num_classes, num_classes))
    i = 0
    gt_id_set = set()
    for key in sorted(gt_dict.keys()):
        gt_path = gt_dict[key]
        result_path = result_dict[key]
        gt = np.asarray(Image.open(gt_path, 'r'))
        gt_id_set.update(np.unique(gt).tolist())
        prediction = np.asanyarray(Image.open(result_path, 'r'))
        hist += fast_hist(gt.flatten(), prediction.flatten(), num_classes)
        i += 1
        if i % 100 == 0:
            print('Finished', i, per_class_iu(hist) * 100)
    gt_id_set.remove([255])
    print('GT id set', gt_id_set)
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])

    print('{:.2f}'.format(miou),
          ', '.join(['{:.2f}'.format(n) for n in list(ious)]))


def evaluate_drivable(gt_dir, result_dir):
    evaluate_segmentation(gt_dir, result_dir, 3, 17)


def main():
    args = parse_args()

    if args.task == 'drivable':
        evaluate_drivable(args.gt, args.result)

if __name__ == '__main__':
    main()
