"""Utility functions for IOTA

"""
from collections import Counter
import numpy as np
import os
import pickle
import pandas as pd

from utils.parsing_utils import create_param_string, blue, green




# Map { image id --> url }
def image_to_url(images_path):
    urls = pd.read_csv(images_path)
    id_url = dict(zip(urls.ImageID, urls.Thumbnail300KURL))
    return id_url


# Parse a DF into a dict {image -> associated labels}
def image_to_labels(annotations):
    img_to_labels, col_name = dict(), 'ImageID'
    images = annotations[col_name].unique().tolist()
    for i in images:
        img_to_labels[i] = annotations[annotations[col_name] == i][
            'LabelName'].values.tolist()
    return image_to_labels


def parse_iota10K(iota, min_rater_count=2):
    gt = dict()
    images = list(set(iota.ImageID.values.tolist()))
    print('IOTA-10K evaluation set has [%d] images' % len(images))
    for im in images:
        gt[im] = iota.loc[iota.ImageID == im, 'LabelName'].tolist()
    gt_majority = majority_vote(gt, min_rater_count)
    return gt_majority, gt


# Choose a single GT label by majority.
# I/O: {image -> [L1, L2, L3]} / {image -> L}.
def majority_vote(gt, min_rater_count):
    gt_majority = dict()
    for i, l in gt.items():
        c = Counter(l)
        labels = [k for k, v in c.items() if v >= min_rater_count and '/m/'
                  in k]
        if not labels: continue
        # In case of tie - choose at random.
        gt_label = np.random.choice(labels) if len(labels) > 1 else ''.join(
            labels)
        gt_majority[i] = gt_label
    return gt_majority


# Load dictionary mapping { id -> ground truth (BaC) label }
def load_evaluation_set(hp, eval_path, gt_dict_path, min_rater_count):
    if os.path.isfile(gt_dict_path):
        print('Load ground-truth dictionary from [%s]' % blue(gt_dict_path))
        image_to_gt = pickle.load(open(gt_dict_path, 'r'))
        return image_to_gt
    df = pd.read_csv(eval_path)
    image_to_sl, image_to_ml = parse_iota10K(df, min_rater_count)
    image_to_gt = image_to_sl if (hp['eval_method'] == 'sl') \
        else image_to_ml
    pickle.dump(image_to_gt, open(gt_dict_path, 'w'))
    print('Saved evaluation-set dictionary to [%s]' % green(gt_dict_path))
    return image_to_gt


# Leave one out to compute raters precision.
# I/O: { image -> [l1,l2,l3] } / image metrics.
def raters_performance(images, gt_path):
    gt_path = gt_path.replace('_sl', '_ml')
    image_to_gt_labels = pickle.load(open(gt_path, 'r'))
    # Ground truth subset.
    for k, v in image_to_gt_labels.items():
        if k not in images: del image_to_gt_labels[k]
    print('Compute raters agreement.')
    image_to_p = dict()
    for k, v in image_to_gt_labels.items():
        agreement = Counter(v).values()  # image -> [3] / [2,1] / [1,2]
        if len(agreement) == 1: image_to_p[k] = 1
        if len(agreement) == 2: image_to_p[k] = 1.0 / 3.0
    print(len(image_to_p))
    p = sum(image_to_p.values()) / len(image_to_p)
    return p


def save_results(hp, results_dir, precision, sem_p, recall, sem_r):
    add_eval_set_to_string = True
    param_string = create_param_string(hp, add_eval_set_to_string)

    path = results_dir + param_string
    if not os.path.exists(path):
        os.makedirs(path)

    output_filename = '/precision_recall.pkl'
    pickle.dump((precision, sem_p, recall, sem_r), open(path +
                                                        output_filename, 'w'))


# Load train, test, validation image - url files into df.
def load_urls_to_df(path_train, path_val, path_test):
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)
    urls = pd.concat([df_train, df_val, df_test])
    urls.set_index('ImageID', inplace=True)
    return urls
