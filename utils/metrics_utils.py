import os
import pickle

import numpy as np
import pandas as pd

import model
from utils.parsing_utils import blue, red, green, create_param_string, timestamp

# Map a label mid to its display name
def load_display_names(classes_filename):
    classes = pd.read_csv(classes_filename, names=['mid', 'name'])
    display_names = dict(zip(classes.mid, classes.name))
    return display_names


#TODO:main function
def compute_metrics(ind_to_label, singles, c22_dict, mis_tree_graph,
                    num_images, num_labels, classes_fn, metrics_fn):

    if False and os.path.isfile(metrics_fn):
        print('Load metrics df from [%s]' % blue(metrics_fn))
        metrics_df = pickle.load(open(metrics_fn, 'rb'))
        return metrics_df

    print('Compute metrics (missing [%s])' % red(metrics_fn))
    display_names = load_display_names(classes_fn)
    metrics = pd.DataFrame.from_dict(ind_to_label, 'index',
                                     columns=['LabelName'])
    metrics.index.name = 'index'
    metrics['DisplayName'] = metrics['LabelName'].apply(lambda x: display_names[x] if x in display_names.keys() else '')
    metrics['px'] = metrics['LabelName'].apply(
        lambda x: float(singles[x]) / num_images)
    metrics['singleton'] = metrics['px'].apply(
        lambda x: model.entropy(np.array([1.0 - x, x])))

    # =============================================
    # Compute delta entropy and DKLs given labels.
    # =============================================
    for start in range(num_labels):  # Run over labels in vocabulary.
        #TODO: here we compute DKL for each label
        sorted_tree = model.sort_graph(mis_tree_graph, start)
        root_p = float(singles[ind_to_label[start]]) / num_images
        root_marginals = np.array([1.0 - root_p, root_p])
        root_dkl = model.compute_dkl(np.array([0.0, 1.0]), root_marginals)
        root_marginals = np.array([0.0, 1.0])
        H, _, Hm, dkl = model.update_marginals(sorted_tree, c22_dict,
                                               start, ind_to_label,
                                               root_marginals)
        if start % 100 == 0:
            print('label=%3d/%d H=%4.2f dkl=%4.2f root_dkl=%4.2f' %
                  (start, num_labels, H, dkl, root_dkl))
        metrics.loc[start, 'H'] = H
        metrics.loc[start, 'dkl'] = dkl + root_dkl

    # =======================================================================
    # Compute information about image distribution MI(T; Image) is simply
    # computed from the fraction of images that contain a label.
    # =======================================================================
    singles_mi = model.compute_image_mi(singles, num_images)
    df = pd.DataFrame.from_dict(singles_mi, 'index', columns=['mi'])
    df.index.name = 'LabelName'
    metrics_df = pd.merge(metrics, df, how='outer', left_on='LabelName',
                          right_index=True)
    add_metric_random(metrics_df)

    # Save to file and return
    path = 'Data/models'
    if not os.path.exists(path): os.makedirs(path)
    pickle.dump(metrics_df, open(metrics_fn, 'wb'))
    print('Saved metrics df to [%s]' % green(metrics_fn))
    return metrics_df


# Compute metrics (over the mixture model)
def compute_metrics_for_tree_mix(models, hp, files):
    param_string = create_param_string(hp, True)
    avg_label_metrics_fn = '%s%s/avg_label_metrics.pkl' % \
                           (files['results_dir'], param_string)
    if os.path.isfile(avg_label_metrics_fn):
        metrics_df = pickle.load(open(avg_label_metrics_fn, 'r'))
    else:
        metrics_per_seed = dict()
        for seed in range(len(models)):
            metrics_per_seed[seed] = models[seed]['label_metrics']
        metrics_df = average_metrics(metrics_per_seed)
        pickle.dump(metrics_df, open(avg_label_metrics_fn, 'w'))
    return metrics_df


# Average the scores over multiple seeds.
def average_metrics(metrics_per_seed):
    metrics_df = []
    for seed in range(len(metrics_per_seed)):
        metrics = metrics_per_seed[seed]
        metrics.set_index('LabelName', inplace=True)
        metrics.drop(columns='DisplayName', inplace=True)
        metrics_df.append(metrics)
    return sum(metrics_df) / (seed + 1)


# Add a random metric (if already exists, re-assign its values)
def add_metric_random(metrics_df):
    num_rows = metrics_df.shape[0]
    metrics_df['random'] = np.random.random(num_rows)
    return metrics_df


# Compute image-dependent metrics, based on label metrics and confidence.
# I/O: Construct a data-frame of the evaluation-set images with the models.
# Headers: image | label | px | singleton | H | dkl | mi | random | wH | y_true
def compute_image_metrics(image_to_gt, label_metrics, files, method='sl',
                          gt_in_voc=True, do_verify=False, y_force=True):
    # Load image-metrics.
    output_fn, eval_fn = files['image_metrics_fn'], files['eval_fn']
    if os.path.isfile(output_fn):
        print('Load image metrics [%s].' % blue(output_fn))
        return pickle.load(open(output_fn, 'r'))
    print('Missing [%s], compute.' % red(output_fn))

    # Read annotations to DF and match labels with its metrics.
    raw_ann = pd.read_csv(eval_fn)
    gtdf = pd.merge(raw_ann[['ImageID', 'LabelName', 'Confidence']],
                    label_metrics, how='left', on='LabelName')

    # Leave only images with GT.
    gtdf = gtdf[gtdf['ImageID'].isin(image_to_gt.keys())]

    # Leave only verified predictions.
    if do_verify and 'oid' in eval_fn:
        v = pd.read_csv(files['ver']).drop(columns=[
            'Source']).rename(columns={'Confidence': 'Verification'})
        gtdf = pd.merge(gtdf, v, on=['ImageID', 'LabelName'])
        gtdf = gtdf[gtdf['Verification'].isin([1])]

    # Compute image dependent metrics.
    gtdf.loc[:, 'cH'] = gtdf.Confidence * gtdf.H
    gtdf.loc[:, 'cDKL'] = gtdf.Confidence * gtdf.dkl
    gtdf.loc[:, 'cMI'] = gtdf.Confidence * gtdf.mi
    gtdf.loc[:, 'cPX'] = gtdf.Confidence * gtdf.px
    gtdf.loc[:, 'cSingleton'] = gtdf.Confidence * gtdf.singleton

    # Leave only labels in vocabulary.
    if gt_in_voc: gtdf = gtdf.dropna(how='any').copy(deep=True)

    disp = load_display_names(files['class_descriptions'])
    for k, v in disp.items(): disp[k] = v.lower()
    gtdf['DisplayName'] = gtdf.LabelName.apply(lambda x: disp[x] if x in
                                                                    disp.keys() else 'none')
    gtdf.loc[:, 'y_true'] = [0.0] * gtdf.shape[0]

    # Multiple labels.
    if method == 'ml':
        gtdf.loc[:, 'R1'] = gtdf.ImageID.apply(lambda x: image_to_gt[x][0])
        gtdf.loc[:, 'R2'] = gtdf.ImageID.apply(lambda x: image_to_gt[x][1])
        gtdf.loc[:, 'R3'] = gtdf.ImageID.apply(lambda x: image_to_gt[x][2])
        gtdf.loc[:, 'R1'] = (gtdf.LabelName == gtdf.R1).astype('int')
        gtdf.loc[:, 'R2'] = (gtdf.LabelName == gtdf.R2).astype('int')
        gtdf.loc[:, 'R3'] = (gtdf.LabelName == gtdf.R3).astype('int')
        gtdf.loc[:, 'y_true'] = gtdf.loc[:, ['R1', 'R2', 'R3']].mean(axis=1)
        gtdf.loc[:, 'y_true'] = gtdf.y_true.apply(lambda x: (x > 0).real)
    # Single Label.
    else:
        gtdf.loc[:, 'y_true'] = gtdf.ImageID.apply(lambda x: image_to_gt[x])
        gtdf.loc[:, 'y_true'] = (gtdf.LabelName == gtdf.y_true).astype(
            'int')

    # Drop images that y_true is not in its returned labels.
    is_gt_returned = gtdf.groupby(['ImageID'])['y_true'].any().to_dict()
    y_false = [k for k in is_gt_returned.keys() if is_gt_returned[k] is False]
    if y_force: gtdf = gtdf[~gtdf['ImageID'].isin(y_false)]

    # Drop rows with labels that are not in vocabulary.
    gtdf = gtdf.dropna(how='any').copy(deep=True)

    # Shuffle to avoid labels with the same confidence sorted the same way.
    indperm = np.random.permutation(gtdf.index)
    gtdf = gtdf.reindex(indperm)
    gtdf = gtdf.sort_values(by='ImageID', ascending=False).reset_index(drop=True)

    # Save results.
    dir = '/'.join(output_fn.split('/')[:-1])
    if not os.path.exists(dir): os.makedirs(dir)
    pickle.dump(gtdf, open(output_fn, 'w'))
    print('Saved metrics to [%s]' % green(output_fn))
    return gtdf


def precision_at_k(k, y_sorted):
    y_true = [0.0] * k
    idx = np.where(y_sorted > 0)[0].tolist()
    for i in idx:
        if i < k: y_true[i] = 1
    y_true = np.array(y_true)
    for c, i in enumerate(idx):
        y_true[i:] = c + 1
    return np.divide(y_true.tolist(), np.arange(1, k + 1, dtype=float))


def recall_at_k(k, y_sorted):
    y_true = [0.0] * k
    idx = np.where(y_sorted > 0)[0].tolist()
    for i in idx:
        if i < k: y_true[i] = 1
    y_true = np.array(y_true)
    for c, i in enumerate(idx):
        y_true[i:] = c + 1
    num_answer = 1 if sum(y_sorted) == 0 else sum(y_sorted)
    return np.divide(y_true.tolist(), float(num_answer))


def compute_recall(df, k):
    print('compute_recall...')
    average_recall, recall_mat, sem_rk = dict(), dict(), dict()
    jpgs = df.ImageID.unique()
    headers = df.columns.tolist()
    remove = ['ImageID', 'LabelName', 'DisplayName', 'y_true',
              'Verification', 'label_count', 'R1', 'R2', 'R3']
    metrics = [val for val in headers if val not in remove]

    n = len(jpgs)
    for metric in metrics:
        average_recall[metric] = np.zeros((1, k))
        recall_mat[metric] = np.zeros(shape=(n, k))

    # r@k is averaged over images.
    i = 0
    for index, image in enumerate(jpgs):
        if index % 1000 == 0: print(timestamp(index, len(jpgs)))
        image_df = df.loc[(df.ImageID == image), :]
        for metric in metrics:
            image_sorted_df = image_df[['DisplayName', metric, 'y_true']]
            image_sorted_df = image_sorted_df.sort_values(by=metric,
                                                          ascending=False)
            y = image_sorted_df['y_true'].values
            r = recall_at_k(k, y)
            recall_mat[metric][i] = r
            average_recall[metric] = np.add(average_recall[metric], np.array(r))
        i = i + 1
    for metric in metrics:
        average_recall[metric] = np.divide(average_recall[metric], n)
        sem_rk[metric] = np.divide(np.std(recall_mat[metric], axis=0),
                                   np.sqrt(n))
    return average_recall, sem_rk, recall_mat


def compute_precision(df, k):
    print('compute_precision...')
    average_precision, precision_mat, sem_pk = dict(), dict(), dict()

    jpgs = df.ImageID.unique()
    headers = df.columns.tolist()
    remove = ['ImageID', 'LabelName', 'DisplayName', 'y_true',
              'Verification', 'label_count', 'R1', 'R2', 'R3']
    metrics = [val for val in headers if val not in remove]

    num_images = len(jpgs)
    for metric in metrics:
        average_precision[metric] = np.zeros((1, k))
        precision_mat[metric] = np.zeros(shape=(num_images, k))

    # P@k is averaged over images.
    for i, image in enumerate(jpgs):
        if i % 1000 == 0: print(timestamp(i, len(jpgs)))
        image_df = df.loc[(df.ImageID == image), :]
        # This code again does many heavy searches ==> optimize.
        for metric in metrics:
            image_sorted_df = image_df[['DisplayName', metric, 'y_true']]
            image_sorted_df = image_sorted_df.sort_values(by=metric,
                                                          ascending=False)
            y = image_sorted_df['y_true'].values
            p = precision_at_k(k, y)
            precision_mat[metric][i] = p
            average_precision[metric] = np.add(average_precision[metric],
                                               np.array(p))
    for metric in metrics:
        average_precision[metric] = np.divide(average_precision[metric],
                                              num_images)
        sem_pk[metric] = np.divide(np.std(precision_mat[metric], axis=0),
                                   np.sqrt(num_images))
    return average_precision, sem_pk, precision_mat
