"""Functions for IOTA probabilistic modelling

"""
import argparse
from collections import OrderedDict
from collections import Counter
import itertools
import numpy as np
import os
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from termcolor import colored
import utils
np.set_printoptions(precision=4)


def set_pair_key(father, son, ind_to_label):
    return ind_to_label[son] + ':' + ind_to_label[father]


# Count labels across all images.
def compute_label_count(ann_data, at_least):
    print('Computing count from  = ' + ann_data)
    oid = pd.read_csv(ann_data)
    labels = oid.LabelName.values.tolist()
    print('Data has ' + str(len(set(labels))) + ' unique labels')
    c = Counter(labels)
    singles = {key: value for key, value in c.items() if value >= at_least}
    num_labels = len(singles)
    print('Data has ' + str(num_labels) + ' labels with count > ' + str(at_least))
    return singles, num_labels


# Mapping {label->ind} & {ind->label}.
def create_label_dict(singles):
    index = range(0, len(singles.keys()))
    labels = singles.keys()
    ind_to_label = dict(zip(index, labels))
    label_to_ind = dict(zip(labels, index))
    return label_to_ind, ind_to_label


def mi_of_count(c):
    pxy = c.astype(float) / np.sum(c)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    pxy_indep = np.outer(px, py)
    p_numerator = pxy[np.nonzero(pxy)]
    p_denom = pxy_indep[np.nonzero(pxy)]
    return sum(p_numerator * np.log2(p_numerator / p_denom))


# To get Maximum spanning tree, calculate -mi instead of mi.
# @ mis - label:label -> mi[float]
def compute_negative_mi(c22, label_to_ind):
    print("Compute MI for all label pairs...")
    mis = dict()
    for l1, l2 in itertools.product(label_to_ind.keys(), repeat=2):
        pair = l1 + ":" + l2
        mis[pair] = 0.0 if l1 == l2 else - mi_of_count(c22[pair])
    return mis


# Returns two dictionaries:
# 1) mis_dict: label:label -> float
# 2) c22_dict: son:father -> 2x2 count matrix
#              The 2x2 matrix has SON on rows, FATHER on columns.
def compute_negative_mi_from_count(label_to_ind, c22_dict):
    print("Compute MI for all label pairs")
    mis_dict = dict()
    for l1, l2 in itertools.product(label_to_ind.keys(), repeat=2):
        pair_key = l1 + ":" + l2
        if l1 == l2:
            mis_dict[pair_key] = 0.0
        else:
            mis_dict[pair_key] = - mi_of_count(c22_dict[pair_key])
    return mis_dict


# Returns a NP array of size (numlabels x numlabels x 2 x 2). Also
# returns the pair distribution of labels, how many images_path contain
# each label pair, its is a dictionary label:label->count.
#
# c22_dict: son:father -> 2x2 count matrix
#           The 2x2 matrix has SON on rows, FATHER on columns.
def compute_label_cooccurrence(filename, label_to_ind, ind_to_label,
                              singles, skip_probability=0.0):
    print("Read label co-ocurrence from [%s] " % colored(filename, 'blue'))
    last_image_id = 'none'
    label_ind_set = set()
    label_names = label_to_ind.keys()
    num_labels = len(label_names)
    count = np.zeros((num_labels, num_labels, 2, 2))
    pair_ind_count = dict()
    print('Shape of count = ' + str(count.shape))
    print('Skip probability %s ' % colored(str(skip_probability), 'green'))

    num_images = 0

    df = pd.read_csv(filename)
    for i_line, row in df.iterrows():
        # Omit lines randomly for bootstrapping.
        if np.random.sample() < skip_probability: continue
        if i_line % 100000 == 0:
            print('line=%d %s' % (i_line, utils.timestamp(i_line)))
        # The line format is: ImageID,Source,LabelName,Confidence
        image_id = row['ImageID']
        new_label = row['LabelName']
        if i_line == 1 or i_line == 0: last_image_id = image_id
        if image_id == last_image_id:
            # Extend the label set
            if new_label in label_to_ind:
                label_ind_set.add(label_to_ind[new_label])
        else:
            # New image. Add counts of all previous lines of the prev image
            num_images += 1
            count, pair_ind_count, label_ind_set = update_counts(
                label_ind_set, count, pair_ind_count,
                label_to_ind, new_label)
            last_image_id = image_id
    num_images += 1
    count, pair_ind_count, label_ind_set = update_counts(
        label_ind_set, count, pair_ind_count, label_to_ind,
        'end-of-file')

    print('Loop over label_pairs, num_labels = ' + str(num_labels))
    pair_count, c22_dict = dict(), dict()
    for pair in itertools.product(range(0, num_labels), repeat=2):
        l1, l2 = pair[0], pair[1]
        if l1*num_labels+l2 % 1000 == 0:
            print(l1*num_labels+l2, utils.timestamp(l1*num_labels+l2))
        c1, c2 = singles[ind_to_label[l1]], singles[ind_to_label[l2]]
        both = count[l1][l2][1][1]
        count[l1][l2][0][1] = c2 - both
        count[l1][l2][1][0] = c1 - both
        count[l1][l2][0][0] = num_images - (c1 + c2 - both)
        pair_name = ind_to_label[l1] + ':' + ind_to_label[l2]
        pair_count[pair_name] = count[l1][l2][1][1]
        c22_dict[pair_name] = count[l1, l2, :, :]

    return count, pair_count, c22_dict, num_images


def update_counts(label_ind_set, count,
                  pair_ind_count, label_to_ind, new_label):

    for il1, il2 in itertools.product(label_ind_set, label_ind_set):
        count[il1, il2, 1, 1] += 1
        pair_ind_key = il1 * 100000 + il2
        pair_ind_count[pair_ind_key] = pair_ind_count.setdefault(
            pair_ind_key, 0) + 1

    # Start a new label set.
    if new_label in label_to_ind:
        label_ind_set = set([label_to_ind[new_label]])
    else:
        label_ind_set = set()

    return count, pair_ind_count, label_ind_set


# Load labels and counts.
def load_count_data(filename, oid_data, hp):
    if os.path.isfile(filename):
        print('Load labels and counts from: [%s]' % colored(filename, 'blue'))
        (count, pair_count, mis_dict, c22_dict, num_images, label_to_ind,
         ind_to_label, singles, num_labels) = pickle.load(open(filename, 'rb'))
    else:
        print(colored("   File [" + filename + "] missing: Compute counts.",
                      "red"))
        singles, num_labels = compute_label_count(oid_data, hp['atleast'])
        label_to_ind, ind_to_label = create_label_dict(singles)
        count, pair_count, c22_dict, num_images = compute_label_cooccurrence(
            oid_data, label_to_ind,  ind_to_label, singles,
            hp['skip_probability'])
        mis_dict = compute_negative_mi_from_count(label_to_ind, c22_dict)

        pickle.dump([count, pair_count, mis_dict, c22_dict,
                     num_images, label_to_ind, ind_to_label,
                     singles, num_labels], open(filename, 'wb'))

    path = 'Data/counts'
    if not os.path.exists(path): os.makedirs(path)
    print('     Found ' + str(len(label_to_ind.keys())) + ' labels')
    return count, pair_count, mis_dict, c22_dict, num_images, label_to_ind, \
           ind_to_label, singles, num_labels


# Print values with abs(mi) > 0.1
def print_top_mis(labels, display_names, mis_dict, c22_dict):
    template = ('{p:>20s} {c00:02d},{c10:02d},{c01:02d},' +
                '{c11:02d} {mi:6.4f}, {nn1:>10s} {nn2}')
    for l1, l2 in itertools.product(labels, labels):
        pair = l1 + ":" + l2
        if pair in c22_dict.keys():
            if abs(mis_dict[pair]) > 0.1:
                c = c22_dict[pair]
                n1 = ''
                n2 = ''
                if l1 in display_names: n1 = display_names[l1]
                if l2 in display_names: n2 = display_names[l2]
                prt = template.format(p=pair, c00=c[0][0], c01=c[0][1],
                                      c10=c[1][0], c11=c[1][1],
                                      mi=mis_dict[pair], nn1=n1, nn2=n2)
                print(prt)


def dict_to_sparse(mis_dict, label_to_ind):
    num_labels = len(label_to_ind)
    mis_dense = np.ndarray(shape=(num_labels, num_labels), dtype=float)
    for l1, l2 in itertools.product(label_to_ind.keys(), repeat=2):
        pair = l1 + ":" + l2
        mis_dense[label_to_ind[l1]][label_to_ind[l2]] = mis_dict[pair]
    mis_sparse = csr_matrix(mis_dense)
    return mis_sparse


# Compute conditional on column, p(row|col) from p(row,col). c22 is the
# count(SON, FATHER).
#
# Returns 2x2 matrix where its first column is p(row|col=0) and
# second column is p(row|col=1)
def joint_to_cond_on_column(c22, tol=0):
    c22 = c22 + tol
    pxy = c22.astype(float) / np.sum(c22)
    py = pxy.sum(axis=0)
    cond22 = np.zeros((2, 2))
    cond22[0][0] = pxy[0][0] / py[0]
    cond22[0][1] = pxy[0][1] / py[1]
    cond22[1][0] = pxy[1][0] / py[0]
    cond22[1][1] = pxy[1][1] / py[1]
    return cond22


# Compute conditional on row, p(col|row) from p(row,col). c22 is the
# count(X,Y) Returns 2x2 matrix where its first row is p(col|row=0) and
# second row is p(col|row=1)
def joint_to_cond_on_row(c22, tol=0):
    p = joint_to_cond_on_column(np.transpose(c22), tol)
    return np.transpose(p)


# Compute p(row,col) from p(row|col) and P(col)
def conditional_to_joint(cond22, py):
    pxy = np.zeros((2, 2))
    pxy[0][0] = cond22[0][0] * py[0]
    pxy[0][1] = cond22[0][1] * py[1]
    pxy[1][0] = cond22[1][0] * py[0]
    pxy[1][1] = cond22[1][1] * py[1]
    return pxy


# Compute p(Y) from p(X,Y) (marginal of father)
def margin_y(c22):
    pxy = c22.astype(float) / np.sum(c22)
    return pxy.sum(axis=0)


# Compute p(X) from p(Y,X) (marginal of son)
def margin_x(c22):
    pxy = c22.astype(float) / np.sum(c22)
    return pxy.sum(axis=1)


# For each node in the sorted graph, computes the CPT { x | pa(x) : cpd 2x2 }
# Returns:
# (1) graph_cpts - dictionary of (son,parent) -> cpt
# (2) graph_counts - dictionary of (son,parent) -> 2x2 count
# (3) index - dictionary of node_index -> node_mid, node_name
# (4) graph_pairs_dict - dict of (son, parent) -> [ mids, names, c22, cpt22 ]
def get_graph_info(c22_dict, sorted_graph, ind_to_label, class_descriptions):
    graph_cpts = OrderedDict()
    graph_counts = OrderedDict()
    index = OrderedDict()
    graph_pairs_dict = OrderedDict()
    names = utils.load_display_names(class_descriptions)
    for father, sons in sorted_graph.items():
        index[father] = [ind_to_label[father], names[ind_to_label[father]]]
        for son in sons:
            pair_key = set_pair_key(father, son, ind_to_label)
            graph_cpts[(son, father)] = joint_to_cond_on_column(
                c22_dict[pair_key])
            graph_counts[(son, father)] = c22_dict[pair_key]

            graph_pairs_dict[(son, father)] = \
                [pair_key, names[ind_to_label[son]] + ":" + names[
                    ind_to_label[father]], graph_counts[(son, father)], \
                 graph_cpts[(son, father)]]
    return graph_cpts, graph_counts, index, graph_pairs_dict


# Propagate through CPD. Compute p(Y) from P(Y|X) and P(X)
def propagate_margin_x(cond22, px):
    pxy = conditional_to_joint(cond22, px)
    return margin_x(pxy), pxy


def entropy(px):
    p_valid = px[np.nonzero(px)]
    p_valid = p_valid / np.sum(p_valid)
    return -sum(p_valid * np.log2(p_valid))


def compute_dkl(px, qx):
    p_valid = px[np.nonzero(px)]
    p_valid = p_valid / np.sum(p_valid)
    q_valid = qx[np.nonzero(px)]
    # assert(np.min(qx) > 0), "{} ".format(q_valid) + "{} ".format(p_valid)
    return sum(p_valid * np.log2(p_valid / q_valid))


def compute_conditional_dkl(px_given_y, qx_given_y, py):
    dkl_x_given_y0 = compute_dkl(px_given_y[:, 0], qx_given_y[:, 0])
    dkl_x_given_y1 = compute_dkl(px_given_y[:, 1], qx_given_y[:, 1])
    if np.isclose(py[0], 0): return dkl_x_given_y1
    if np.isclose(py[1], 0): return dkl_x_given_y0
    print("{} ".format(qx_given_y[:, 0]) + "{} ".format(px_given_y[:, 0]))
    print("{} ".format(qx_given_y[:, 1]) + "{} ".format(px_given_y[:, 1]))
    print("py0=%8.6f dkl|y0=%8.6f py1=%8.6f dkl|y1=%8.6f" % (
        py[0], dkl_x_given_y0, py[1], dkl_x_given_y1))
    return dkl_x_given_y0 * py[0] + dkl_x_given_y1 * py[1]


# Compute the entropy H(x|pa(x)) of x_i given a single parent pa(xi)
# Expect a joint 2x2 count, returns a scalar
def compute_pair_conditional_entropy(c22):
    py = margin_y(c22)
    px_given_y = joint_to_cond_on_column(c22, 0.000000001)
    h_x_given_y0 = entropy(px_given_y[:, 0])
    h_x_given_y1 = entropy(px_given_y[:, 1])
    if np.isclose(py[0], 0): return h_x_given_y1
    if np.isclose(py[1], 0): return h_x_given_y0
    return h_x_given_y0 * py[0] + h_x_given_y1 * py[1]


def compute_tree_entropy(sorted_graph, start, c22_dict, singles, num_images,
                         ind_to_label, debug=False):
    # sorted_graph assumes that every node has a single father,
    # and multiple sons held as a set.
    print('Compute tree entropy.')
    p_root = float(singles[ind_to_label[start]]) / num_images
    H = entropy(np.array([p_root, 1.0 - p_root]))
    if debug: print('proot=' + str(p_root),
                    'n=' + str(num_images),
                    'psingles=', singles[ind_to_label[start]],
                    start, 'H = ' + str(H))
    for father, sons in sorted_graph.items():
        for son in sons:
            pair_key = set_pair_key(father, son, ind_to_label)
            c22 = c22_dict[pair_key]
            h = compute_pair_conditional_entropy(c22)
            H += h
            if debug: print('father=', father, 'son=', son, c22.tolist(), h)
    print("   Tree Entropy = %8.6f" % H)
    return H


# Translate a graph represented as a csr matrix to a graph represented
# as <key -> set>
def csr_to_graph(csr_matrix):
    graph = dict()
    fathers, sons = csr_matrix.nonzero()
    for father, son in zip(fathers, sons):
        if father not in graph:
            graph[father] = set([son])
        else:
            graph[father].add(son)
    # Add the reverse edges as well
    for son, father in zip(fathers, sons):
        if father not in graph:
            graph[father] = set([son])
        else:
            graph[father].add(son)
    return graph


# Reorder a directed graph in the form of a sparse csr_matrix, such
# that each node has a single parent only.
def sort_graph(graph, start):
    # Start from 'root', perform DFS, treating graph as undirected.
    sorted_graph = dict()
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
            sorted_graph[vertex] = graph[vertex] - visited
    return sorted_graph


# Expecting a dictionary as input, that contains either a singles-count
# dict, or a pair-count dict. The mutual information is the difference
# of entropies, which is simply log2(num_images) - log2(count)
def compute_image_mi(count_dict, num_images):
    mi = dict()
    for label, count in count_dict.items():
        lls = label.split(':')
        if len(lls) == 2:
            # skip mi for same-label pairs
            if lls[0] == lls[1]: continue
        mi[label] = np.log2(num_images) - np.log2(count)
    return mi


def singles_count_to_marginals(singles, label_to_ind, num_images):
    marginals = dict()
    for key, value in singles.items():
        marginals[label_to_ind[key]] = float(value) / float(num_images)
    return marginals


# Sanity check that conditional DKLs are zero
def check_dkl(pxy, p_cpd, new_marginal_father, c22_dict, pair_key):
    q_cpd = joint_to_cond_on_column(pxy)
    dkl_edge = compute_conditional_dkl(q_cpd, p_cpd,
                                       new_marginal_father)
    dkl_edge = np.round(dkl_edge, 10)
    message = ("\n\np_cpd = " +
               "{}".format(p_cpd) +
               "\nq_cpd = " +
               "{}".format(q_cpd) +
               "\nmarginals[father] = " +
               "{}".format(new_marginal_father) +
               "\nc22_dict = " +
               "{}".format(c22_dict[pair_key]) +
               "\ndkl = " +
               "{}".format(dkl_edge))
    assert np.isfinite(dkl_edge), message
    assert np.isclose(dkl_edge, 0.0), message


# Propagate marginals through the graph from a root node. Compute the
# new conditional entropy while traversing the graph. The root label
# (root_ind) is true.
# Implementation:
# 1. Travers the tree in DFS
# 2. Use father marginal to compute P(son,father) = P(son|father)P(father)
# 3. compute the marginal of the son
# 4. propagate to its descendants
#
# Returns:
# H:             Entropy of the tree with the new marginals.
# new_marginals: The updated marginals.
# marginal_H:    The sum of entropies of the marginals.
# dkl:           Dkl(new dist || original dist)
def update_marginals(sorted_tree_graph, c22_dict, root_ind,
                     ind_to_label, root_marginals):
    visited, stack = set(), [root_ind]
    new_marginals = dict()
    new_marginals[root_ind] = root_marginals
    H = entropy(np.array(root_marginals))
    marginal_H = H
    dkl = 0.0
    while stack:
        father = stack.pop()
        for son in sorted_tree_graph[father]:
            stack.extend([son])
            pair_key = set_pair_key(father, son, ind_to_label)
            old_marginal = np.array(margin_x(c22_dict[pair_key]))
            p_cpd = joint_to_cond_on_column(c22_dict[pair_key])
            new_marginals[son], pxy = propagate_margin_x(p_cpd,
                                                         new_marginals[father])
            dkl_marginal = compute_dkl(new_marginals[son], old_marginal)
            H += compute_pair_conditional_entropy(pxy)
            marginal_H += entropy(new_marginals[son])
            dkl += dkl_marginal
            if False: check_dkl(pxy, p_cpd, new_marginals[father],
                                c22_dict, pair_key)
    return H, new_marginals, marginal_H, dkl
