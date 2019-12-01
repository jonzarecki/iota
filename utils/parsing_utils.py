import argparse
import json
import os
from datetime import datetime

import numpy as np
from termcolor import colored

np.set_printoptions(precision=4)


# Handle flags.
def parse_flags():
    parser = argparse.ArgumentParser(description='Implement IOTA.')

    # Specify the distribution on which we learn the metrics.
    parser.add_argument('--annotations', default='oid',
                        help='allowed values: oid')
    parser.add_argument('--skip_probability', default=0.1,
                        help='probability to skip a label in the data')
    parser.add_argument('--atleast', default=100,
                        help='threshold on label occurrence')
    parser.add_argument('--rater', default='machine',
                        help='allowed values: human, machine')
    parser.add_argument('--split', default='validation',
                        help='allowed values: train, validation, test')
    parser.add_argument('--max_seed', default=10,
                        help='enum over seeds 0,...,max_seed')
    parser.add_argument('--eval_set', default='iota_validation',
                        help='allowed values: iota, iota_validation, iota_test')
    parser.add_argument('--min_rater_count', default=2,
                        help='only consider ratings with that many raters agree')
    parser.add_argument('--plot_figures', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='if true, plot and save figures')
    parser.add_argument('--eval_method', default='sl',
                        help='allowed values: sl (single-label), '
                             'ml(multi-label)')
    parser.add_argument('--k', default=10,
                        help='precision@k value')

    # The following flags are implemented in compute_image_metrics().
    parser.add_argument('--do_verify', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Use verified labels if true')
    parser.add_argument('--y_force', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Consider only cases in which GT label'
                             ' is in the returned labels')
    parser.add_argument('--gt_vocab', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='When false, also includes images whose GT '
                             'not in vocabulary')
    args = parser.parse_args()

    assert (args.split in ['train', 'validation', 'test'])
    assert (args.rater in ['machine', 'human'])
    assert (args.min_rater_count > 0)
    assert (int(args.min_rater_count) < 4)
    assert (args.eval_set in ['iota_validation', 'iota_test'])
    assert (args.eval_method in ['sl', 'ml'])
    assert (float(args.skip_probability) < 1.0)
    assert (float(args.skip_probability) >= 0.0)
    assert (args.gt_vocab == args.do_verify)
    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_param_string(hp, add_eval_set=False):
    # Global parameters.
    param_string = '_'.join([hp['annotations'], hp['split'], hp['rater'],
                             str(hp['atleast'])])
    param_string += '_s' + str(hp['seed'])
    param_string += 'p' + str(hp['skip_probability'])

    # Add eval-set dependent parameters.
    if add_eval_set:
        param_string += '_' + hp['eval_set']
        param_string += '_ver' if hp['do_verify'] else '_unver'
        param_string += '_voc' if hp['gt_vocab'] else '_novoc'
        param_string += '_yforce' if hp['y_force'] else '_noyforce'
        param_string += '_' + hp['eval_method']
    return param_string


def set_file_path(hp):
    files = dict()
    param_string = create_param_string(hp, False)
    #
    # data
    #   |_ ground_truth
    #   |_oid
    #       |_classes
    #       |_train
    #       |_validation
    #       |_test
    # results
    #   |_models
    #   |_counts
    #
    results_dir = os.environ.get('RES_DIR') if 'RES_DIR' in os.environ else 'results/'
    data_dir = os.environ.get('OID_DIR') if 'OID_DIR' in os.environ else 'data/'

    files['oid_dir'] = data_dir
    files['results_dir'] = results_dir
    files['counts_dir'] = files['results_dir'] + 'counts/'
    files['models_dir'] = files['results_dir'] + 'models/'

    files['annotations']=files['oid_dir']+hp['split']+'/annotations-'+hp['rater']+'.csv'
    files['eval_fn']=files['oid_dir']+hp['split']+'/annotations-'+hp['rater']+'.csv'
    files['ver'] = files['oid_dir'] + hp['split'] + '/annotations-human.csv'
    files['images_path'] = files['oid_dir'] + hp['split'] + '/images.csv'
    files['class_descriptions'] = files['oid_dir'] + 'classes/class-descriptions.csv'

    # Path to IOTA-10K.
    files['iota10k'] = files['oid_dir'] + 'evaluation/' + hp['eval_set'] +'.csv'
    files['gt_dir'] = 'data/ground_truth/'  # pkl

    # Output files.
    files['model_fn'] = '%smodel_%s.pkl' % (files['models_dir'], param_string)
    files['counts_fn'] = '%scount_%s.pkl' % (files['counts_dir'], param_string)
    files['metrics_fn'] = '%slabel_metrics_%s.pkl' % (files['models_dir'],
                                                      param_string)
    files['gt_fn'] = '%sgt_dict_%s_%s.pkl' % (files['gt_dir'],
                                              hp['eval_set'],
                                              hp['eval_method'])

    # Output file names, (depend on eval set)
    param_string_eval_set = create_param_string(hp, True)
    files['image_metrics_fn'] = ('%s%s/image_metrics_%s.pkl' %
                                 (files['results_dir'],
                                  param_string_eval_set,
                                  param_string_eval_set))
    return files


def blue(s): return colored(s, 'blue')


def red(s): return colored(s, 'red')


def green(s): return colored(s, 'green')


def timestamp(index, total=-1):
    stamp = datetime.now().strftime('%H:%M:%S.%f')[:-4]
    if total < 0:
        return stamp
    else:
        return ('index=%d/%d %s' % (index, total, stamp))


def print_top_labels(k, dkls, ind_to_label, display_names):
    inds = np.argsort(dkls)
    for i in range(k):
        ind = inds[i]
        print(str(i) + ' dkl = ' + str(dkls[ind]) +
              '  ' + ind_to_label[ind] +
              '  ' + display_names[ind_to_label[ind]])


def parse_json(filepath):
    root = json.loads(open(filepath).read())
    return walk(root)


# Turn the JSON file to a dictionary {lbl, parent}
def walk(node, res={}):
    if 'children' in dict.keys(node):
        kids_list = node['children']
        for curr in kids_list:
            res.update({curr['name']: node['name']})
            walk(curr)
    else:
        return
    return res


# Map a label mid to its display name
def load_display_names(classes_filename):
    classes = pd.read_csv(classes_filename, names=['mid', 'name'])
    display_names = dict(zip(classes.mid, classes.name))
    return display_names

