
from rusty_axe.utils import fast_knn, double_fast_knn, hacked_louvain, generate_feature_value_html, sister_distance
from rusty_axe.node import Node, Reduction, Filter
from rusty_axe.prediction import Prediction
from rusty_axe.sample_cluster import SampleCluster
from rusty_axe.node_cluster import NodeCluster
from rusty_axe.tree import Tree

from json import dumps as jsn_dumps
from os import makedirs
from shutil import copyfile, rmtree

import re
import json
import sys
import os
import tempfile as tmp
import random
import glob
import pickle
from functools import reduce
from multiprocessing import Pool
import copy
from pathlib import Path

import numpy as np

import scipy.special
from scipy.stats import linregress
from scipy.spatial.distance import jaccard
from scipy.spatial.distance import squareform
from scipy.optimize import nnls
from scipy.cluster import hierarchy as hrc
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import dendrogram, linkage

import sklearn
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge, Lasso

from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100


class Forest:

    def __init__(self, trees, input, output, input_features=None, output_features=None, samples=None, split_labels=None, cache=False):
        if input_features is None:
            input_features = [str(i) for i in range(input.shape[1])]
        if output_features is None:
            output_features = [str(i) for i in range(output.shape[1])]
        if samples is None:
            samples = [str(i) for i in range(input.shape[0])]

        self.cache = cache

        self.truth_dictionary = TruthDictionary(
            output, output_features, samples)

        self.input = input
        self.output = output
        self.samples = samples

        self.input_features = input_features
        self.output_features = output_features

        self.input_dim = input.shape
        self.output_dim = output.shape

        self.trees = trees

        for i, node in enumerate(self.nodes()):
            node.index = i

        if split_labels is not None:
            self.external_split_labels(self.nodes(), split_labels, roots=True)

########################################################################
########################################################################

# Pool methods

########################################################################
########################################################################

    # A method that allows one to summon a multiprocessing pool general to the forest

    def pool(self):
        if hasattr(self, 'pool'):
            if self.pool_object is not None:
                return self.pool_object

        else:
            pool_object = Pool()
            self.pool_object = pool_object
            return self.pool_object

    def release_pool(self):
        self.pool_object.terminate()

    def __del__(self):
        self.release_pool()
        del self

########################################################################
########################################################################

# Node selection methods

# Methods for picking specific nodes from the forest

########################################################################
########################################################################

    def nodes(self, root=True, depth=None):
        nodes = []
        for tree in self.trees:
            nodes.extend(tree.nodes(root=root))
        if depth is not None:
            nodes = [n for n in nodes if n.level <= depth]
        return nodes

    def reindex_nodes(self):
        nodes = self.nodes()
        for i, node in enumerate(nodes):
            node.index = i

    def reindex_samples(self, samples):
        map = {s: i for i, s in list(enumerate(samples))}
        for node in self.nodes():
            if node.local_samples is not None:
                new_samples = [map[s] for s in node.local_samples]
                node.local_samples = new_samples

    def leaves(self, depth=None):
        leaves = []
        for tree in self.trees:
            leaves.extend(tree.leaves(depth=depth))
        return leaves

    def level(self, target):
        level = []
        for tree in self.trees:
            level.extend(tree.level(target))
        return level

    def stems(self, depth=None):
        stems = []
        for tree in self.trees:
            stems.extend(tree.stems())
        if depth is not None:
            stems = [s for s in stems if s.level <= depth]
        return stems

    def roots(self):
        return [tree.root for tree in self.trees]

    def trim(self, limit):

        for i, tree in enumerate(self.trees):
            print(f"Trimming {i}\r", end='')
            tree.trim(limit)
        print("")

        self.reindex_nodes()
        self.reset_cache()

    def leaf_mask(self):
        leaf_mask = np.zeros(len(self.nodes()), dtype=bool)
        leaf_mask[[leaf.index for leaf in self.leaves()]] = True
        return leaf_mask

########################################################################
########################################################################

# NODE/MATRIX METHODS

# Methods for turning a set of nodes into an encoding matrix

# Encoding matrices are boolean matrices, usually node x property
# Sample encoding matrix would be i,j, where i is ith node and j is whether
# sample j appears in that node

########################################################################
########################################################################

    def node_representation(self, nodes=None, mode='additive_mean', metric=None, pca=0):
        from sklearn.decomposition import IncrementalPCA

        # ROWS ARE NODES, COLUMNS ARE WHATEVER

        if nodes is None:
            nodes = self.nodes()

        if mode == 'gain':
            encoding = self.local_gain_matrix(nodes).T
        elif mode == 'error_ratio':
            encoding = self.error_ratio_matrix(nodes).T
        elif mode == 'additive':
            encoding = self.additive_matrix(nodes).T
        elif mode == 'additive_mean' or mode == 'conditional_gain':
            encoding = self.mean_additive_matrix(nodes).T
        elif mode == 'sample':
            encoding = self.node_sample_encoding(nodes).T
        elif mode == 'sister':
            encoding = self.node_sister_encoding(nodes).T
        elif mode == 'median' or mode == 'medians':
            encoding = self.median_matrix(nodes)
        elif mode == 'mean' or mode == 'means':
            encoding = self.mean_matrix(nodes)
        elif mode == 'factor':
            encoding = self.node_factor_encoding(nodes)
        elif mode == 'partial':
            encoding = self.partial_matrix(nodes)
        elif mode == 'partial_absolute':
            encoding = self.partial_absolute(nodes)
        else:
            raise Exception(f"Mode not recognized:{mode}")

        if pca > 0:
            if pca > encoding.shape[1]:
                print("WARNING, PCA DIMENSION TOO SMALL, PICKING MINIMUM")
                pca = np.min([pca, encoding.shape[1]])
            # print(f"debug:{encoding.shape}")
            from sklearn.decomposition import IncrementalPCA
            model = IncrementalPCA(n_components=pca)
            chunks = int(np.floor(encoding.shape[0] / 10000)) + 1
            last_chunk = encoding.shape[0] - ((chunks - 1) * 10000)
            for i in range(1, chunks):
                print(f"Learning chunk {i}\r", end='')
                model.partial_fit(encoding[(i - 1) * 10000:i * 10000])
            model.partial_fit(encoding[-last_chunk:])
            # transformed = model.transform(encoding)
            # print(f"Chunks:{chunks}")
            transformed = np.zeros((encoding.shape[0], pca))
            for i in range(1, chunks):
                print(f"Transforming chunk {i}\r", end='')
                # print(f"coordinates:{((i-1)*10000,i*10000)}")
                transformed[(
                    i - 1) * 10000:i * 10000] = model.transform(encoding[(i - 1) * 10000:i * 10000])
            # print(f"coordinates:{-last_chunk}")
            transformed[-last_chunk:] = model.transform(encoding[-last_chunk:])
            print("")
            encoding = transformed
            # encoding = PCA(n_components=pca).fit_transform(encoding)

            # encoding = PCA(n_components=pca).fit_transform(encoding)

        if metric == "sister":
            if mode != "sister":
                raise Exception(f"Mode and metric mismatched {mode},{metric}")
            else:
                representation = sister_distance(encoding)

        if metric is not None:
            representation = squareform(pdist(encoding, metric=metric))
        else:
            representation = encoding

        return representation

    def agglomerate_representation(representation, feature_metric='correlation', sample_metric='cosine'):
        feature_sort = dendrogram(linkage(
            representation.T, metric=feature_metric, method='average'), no_plot=True)['leaves']
        sample_sort = dendrogram(linkage(
            representation, metric=sample_metric, method='average'), no_plot=True)['leaves']
        return representation[sample_sort].T[feature_sort].T

    def node_sample_encoding(self, nodes):

        # ROWS: SAMPLES
        # COLUMNS: NODES

        encoding = np.zeros((len(self.samples), len(nodes)), dtype=bool)
        for i, node in enumerate(nodes):
            encoding[:, i] = node.encoding()
        return encoding

    def node_factor_encoding(self, nodes):

        encoding = np.zeros((len(nodes), len(self.split_clusters)), dtype=bool)

        for i, factor in enumerate(self.split_clusters):
            mask = factor.node_mask()
            encoding[:, i][mask] = True

        return encoding

    def node_sister_encoding(self, nodes):
        encoding = np.zeros((len(self.samples), len(nodes)), dtype=int)
        for i, node in enumerate(nodes):
            encoding[:, i][node.sample_mask()] = 1
            if node.sister() is not None:
                encoding[:, i][node.sister().sample_mask()] = -1
        return encoding

    def absolute_gain_matrix(self, nodes):
        gains = np.zeros((len(self.output_features), len(nodes)))
        for i, node in enumerate(nodes):
            gains[i] = node.absolute_gains()
        return gains

    def local_gain_matrix(self, nodes):
        gains = np.zeros((len(self.output_features), len(nodes)))
        for i, node in enumerate(nodes):
            gains[:, i] = node.local_gains()
        return gains

    def error_ratio_matrix(self, nodes):
        ratios = np.zeros((len(self.output_features), len(nodes)))
        for i, node in enumerate(nodes):
            ratios[:, i] = node.mean_error_ratio()
        return ratios

    def additive_matrix(self, nodes):
        gains = np.zeros((len(self.output_features), len(nodes)))
        for i, node in enumerate(nodes):
            gains[:, i] = node.additive_gains()
        return gains

    def mean_additive_matrix(self, nodes):
        gains = np.zeros((len(self.output_features), len(nodes)))
        for i, node in enumerate(nodes):
            gains[:, i] = node.additive_mean_gains()
        return gains

    def conditional_gain_matrix(self, nodes):
        return self.mean_additive_matrix(nodes)

    def mean_matrix(self, nodes):
        predictions = np.zeros((len(nodes), len(self.output_features)))
        for i, node in enumerate(nodes):
            predictions[i] = node.means()
        return predictions

    def median_matrix(self, nodes):
        predictions = np.zeros((len(nodes), len(self.output_features)))
        for i, node in enumerate(nodes):
            if i % 10 == 0:
                print(f"Estimating node {i}\r", end='')
            predictions[i] = node.medians()
        return predictions

    def weight_matrix(self, nodes):
        weights = np.zeros((len(nodes), len(self.output_features)))
        for i, node in enumerate(nodes):
            weights[i] = node.weights
        return weights

    def partial_matrix(self, nodes):
        partials = np.zeros((len(nodes), len(self.output_features)))
        for i, node in enumerate(nodes):
            if i % 1000 == 0:
                print(f"Node {i}\r", end='')
            partials[i] = node.partials()
        print("")
        return partials

    def partial_absolute(self, nodes):
        partials = np.zeros((len(nodes), len(self.output_features)))
        for i, node in enumerate(nodes):
            if i % 1000 == 0:
                print(f"Node {i}\r", end='')
            partials[i] = node.absolute_partials()
        print("")
        return partials

    def weighted_prediction_matrix(self, nodes):
        weighted_predictions = np.zeros(
            (len(nodes), len(self.output_features)))
        for i, node in enumerate(nodes):
            weighted_predictions[i] = node.weighted_prediction_cache
        return weighted_predictions

########################################################################
########################################################################

# LOADING/CREATION METHODS

# This section deals with methods that load and unload the forest
# from disk

########################################################################
########################################################################

    def backup(self, location):
        print("Saving forest")

        # We need to let go of the pool to back up to disk
        self.pool_object = None
        print(location)
        try:
            with open(location, mode='bw') as f:
                pickle.dump(self, f)
        except Exception:
            print("Failed to save")

    def load(location):
        with open(location, mode='br') as f:
            return pickle.load(f)

    def load_from_rust(location, prefix="/run", ifh="/run.ifh", ofh='run.ofh', clusters='run.cluster', input="input.counts", output="output.counts"):

        combined_tree_files = sorted(
            glob.glob(location + prefix + "*.compact"))

        input = np.loadtxt(location + input)
        output = np.loadtxt(location + output)
        ifh = np.loadtxt(location + ifh, dtype=str)
        ofh = np.loadtxt(location + ofh, dtype=str)

        split_labels = None
        try:
            print(f"Looking for clusters:{location+clusters}")
            split_labels = np.loadtxt(location + clusters, dtype=int)
        except Exception:
            pass

        first_forest = Forest([], input_features=ifh, output_features=ofh,
                              input=input, output=output, split_labels=split_labels)

        trees = []
        for tree_file in combined_tree_files:
            print(f"Loading {tree_file}\r", end='')
            trees.append(
                Tree(json.load(open(tree_file.strip())), first_forest))

        # first_forest.prototype = Tree(json.load(open(location+prefix+".prototype")),first_forest)

        first_forest.trees = trees

        for i, node in enumerate(first_forest.nodes()):
            node.index = i

        sample_encoding = first_forest.node_sample_encoding(
            first_forest.leaves())

        if np.sum(np.sum(sample_encoding, axis=1) == 0) > 0:
            print("WARNING, UNREPRESENTED SAMPLES")

        return first_forest

    def from_sklearn(forest):

        raw_trees = [e.tree_ for e in forest.estimators_]

        trees = []

        def node_recursion(index, children_left, children_right):
            nodes = []
            left_child = children_left[index]
            right_child = children_right[index]
            if left_child > 0 and right_child > 0:
                nodes.extend(node_recursion(
                    left_child, children_left, children_right))
                nodes.extend(node_recursion(
                    right_child, children_left, children_right))
                nodes.append(left_child)
                nodes.append(right_child)
            return nodes

        for raw_tree in raw_trees:
            children_left = raw_tree.children_left
            children_right = raw_tree.children_right
            nodes = node_recursion(0, children_left, children_right)
            trees.append(nodes)

        return trees

    def from_sklearn(forest):

        raw_trees = [e.tree_ for e in forest.estimators_]

        trees = []

        def node_recursion(index, children_left, children_right):
            nodes = []
            left_child = children_left[index]
            right_child = children_right[index]
            if left_child > 0 and right_child > 0:
                nodes.extend(node_recursion(
                    left_child, children_left, children_right))
                nodes.extend(node_recursion(
                    right_child, children_left, children_right))
                nodes.append(left_child)
                nodes.append(right_child)
            return nodes

        for raw_tree in raw_trees:
            children_left = raw_tree.children_left
            children_right = raw_tree.children_right
            nodes = node_recursion(0, children_left, children_right)
            trees.append(nodes)

        return trees

    def derive_samples(self, samples):

        new_trees = []

        for i, tree in enumerate(self.trees):
            print(f"Deriving tree {i}")
            new_trees.append(tree.derive_samples(samples))

        new_forest = Forest(
            new_trees,
            copy.deepcopy(self.input[samples]),
            copy.deepcopy(self.output[samples]),
            input_features=copy.deepcopy(self.input_features),
            output_features=copy.deepcopy(self.output_features),
            samples=list(np.array(self.samples)[samples]),
            split_labels=None,
            cache=self.cache
        )

        for tree in new_forest.trees:
            tree.forest = new_forest

        for node in new_forest.nodes():
            node.forest = new_forest

        split_labels = [n.split_cluster for n in new_forest.nodes()]
        new_forest.external_split_labels(
            new_forest.nodes(), split_labels, roots=True)

        new_forest.reindex_samples(samples)

        new_forest.reset_cache()

        return new_forest

    def set_cache(self, value):
        self.cache = value
        for node in self.nodes():
            node.cache = value

    def compute_cache(self):
        for i, tree in enumerate(self.trees):
            print(f"Tree {i}", end='\r')
            tree.root.compute_cache()
        print("")

    def reset_cache(self):
        for node in self.nodes():
            node.reset_cache()

    def add_output_feature(self, feature_values, feature_name=None):

        self.reset_cache()

        if not hasattr(self, 'core_output_features'):
            self.core_output_features = len(self.output_features)

        if feature_name is None:
            feature_name = str(len(self.output_features))

        if feature_name in self.truth_dictionary.feature_dictionary.keys():
            raise Exception("REPEAT FEATURE")

        feature_index = len(self.output_features)

        self.output_features = np.concatenate(
            [self.output_features, np.array([feature_name, ])])
        self.output = np.concatenate(
            [self.output, np.array([feature_values, ]).T], axis=1)
        self.truth_dictionary.feature_dictionary[feature_name] = feature_index

        for node in self.nodes():
            node.weights = np.append(node.weights, 1.)

    def remove_output_feature(self, feature):

        if feature not in self.truth_dictionary.feature_dictionary.keys():
            raise Exception("Feature not found")

        self.reset_cache()

        feature_index = self.truth_dictionary.feature_dictionary[feature]

        self.output_features = np.delete(self.output_features, feature_index)
        self.output = np.delete(self.output, feature_index, 1)
        for node in self.nodes():
            node.weights = np.delete(node.weights, feature_index)

        new_feature_dictionary = {f: i for i,
                                  f in enumerate(self.output_features)}
        self.truth_dictionary.feature_dictionary = new_feature_dictionary

    def reset_output_featuers(self):

        if hasattr(self, 'core_output_features'):
            if self.core_output_features < len(self.output_features):
                removed_features = self.output_features[self.core_output_features:]
                print(f"Removing:{removed_features}")
                for feature in removed_features:
                    self.remove_output_feature(feature)

########################################################################
########################################################################

# PREDICTION METHODS

# This section deals with methods that allow predictions on
# samples

########################################################################
########################################################################

    def predict(self, matrix):
        prediction = Prediction(self, matrix)
        return prediction

    def predict_sample_leaves(self, sample):
        sample_leaves = []
        for tree in self.trees:
            sample_leaves.extend(tree.root.predict_sample_leaves(sample))
        return sample_leaves

    def predict_sample_nodes(self, sample):
        sample_nodes = []
        for tree in self.trees:
            sample_nodes.extend(tree.root.predict_sample_nodes(sample))
        return sample_nodes

    def predict_vector_leaves(self, vector, features=None):
        # if features is None:
        # features = self.input_features
        sample = {feature: value for feature,
                  value in zip(range(len(vector)), vector)}
        return self.predict_sample_leaves(sample)

    def predict_vector_nodes(self, vector, features=None):
        # if features is None:
        #     features = self.input_features
        sample = {feature: value for feature,
                  value in zip(range(len(vector)), vector)}
        return self.predict_sample_nodes(sample)

    def predict_node_sample_encoding(self, matrix, leaves=True, depth=None):
        encodings = []
        for i, root in enumerate(self.roots()):
            print(f"Predicting tree:{i}\r", end='')
            encodings.append(root.predict_matrix_encoding(matrix))
            encodings.append(np.ones(matrix.shape[0], dtype=bool))
        print('')
        encoding = np.vstack(encodings)
        if leaves:
            encoding = encoding[self.leaf_mask()]
        if depth is not None:
            depth_mask = np.zeros(encoding.shape[0], dtype=bool)
            for n in self.nodes():
                if n.level <= depth:
                    depth_mask[n.index] = True
            encoding = encoding[depth_mask]
        return encoding

########################################################################
########################################################################

# CLUSTERING METHODS

########################################################################
########################################################################

    def split_labels(self, depth=3):

        nodes = self.nodes(depth=depth)
        return np.array([n.split_cluster for n in nodes])

    def set_sample_labels(self, sample_labels):

        self.sample_labels = np.array(sample_labels).astype(dtype=int)

        cluster_set = set(sample_labels)
        clusters = []
        for cluster in cluster_set:
            samples = np.arange(len(self.sample_labels))[
                self.sample_labels == cluster]
            clusters.append(SampleCluster(self, samples, int(cluster)))

        self.sample_clusters = clusters

        one_hot = np.array([sample_labels == x.id for x in clusters])

        self.sample_cluster_encoding = one_hot

    def cluster_samples_simple(self, override=False, pca=None, resolution=1, **kwargs):

        if hasattr(self, 'sample_labels') and override:
            self.reset_sample_clusters()

        counts = self.output

        if hasattr(self, 'sample_clusters') and not override:
            print("Clustering has already been done")
            return self.sample_labels
        else:
            if pca is not None:
                counts = PCA(n_components=pca).fit_transform(counts)
                self.set_sample_labels(hacked_louvain(
                    fast_knn(counts, **kwargs), resolution=resolution))
            else:
                self.set_sample_labels(hacked_louvain(
                    fast_knn(counts, **kwargs), resolution=resolution))

        return self.sample_labels

    def cluster_samples_encoding(self, override=False, pca=None, depth=None, resolution=1, **kwargs):

        # Todo: remove this hack
        if depth is not None:
            depth_limit = depth

        if hasattr(self, 'sample_labels') and override:
            self.reset_sample_clusters()

        leaves = self.leaves(depth=depth)

        encoding = self.node_sample_encoding(leaves)

        if pca is not None:
            encoding = PCA(n_components=pca).fit_transform(encoding)

        if hasattr(self, 'sample_clusters') and not override:
            print("Clustering has already been done")
        else:
            self.set_sample_labels(
                1 + hacked_louvain(fast_knn(encoding, **kwargs), resolution=resolution))

        return self.sample_labels

    def cluster_samples_leaf_cluster(self, override=False, *args, **kwargs):

        if hasattr(self, 'sample_labels') and override:
            self.reset_sample_clusters()

        leaves = [n for n in self.nodes() if hasattr(n, 'leaf_cluster')]
        encoding = self.node_sample_encoding(leaves)
        leaf_clusters = np.array([leaf.leaf_cluster for leaf in leaves])
        leaf_cluster_sizes = np.array(
            [np.sum(leaf_clusters == cluster) for cluster in range(len(set(leaf_clusters)))])

        print(f"encoding dimensions: {encoding.shape}")

        sample_labels = []

        for leaf_mask in encoding:
            leaf_cluster_counts = np.array(
                [np.sum(leaf_clusters[leaf_mask] == lc) for lc in range(len(leaf_cluster_sizes))])
            odds = leaf_cluster_counts / leaf_cluster_sizes
            sample_labels.append(np.argmax(odds))

        self.set_sample_labels(np.array(sample_labels).astype(dtype=int))

        return self.sample_labels

    def set_leaf_labels(self, labels):

        leaves = self.leaves()
        self.leaf_labels = np.array(labels)

        cluster_set = set(self.leaf_labels)

        clusters = []

        for cluster in cluster_set:
            leaf_index = np.arange(len(self.leaf_labels))[
                self.leaf_labels == cluster]
            clusters.append(NodeCluster(
                self, [leaves[i] for i in leaf_index], cluster))

        self.leaf_clusters = clusters
        for leaf, label in zip(leaves, self.leaf_labels):
            leaf.leaf_cluster = label

    def cluster_leaves_predictions(self, override=False, mode='mean', *args, **kwargs):

        leaves = self.leaves()
        predictions = self.node_representation(leaves, mode=mode)

        if hasattr(self, 'leaf_clusters') and not override:
            print("Clustering has already been done")
            return self.leaf_labels
        else:
            self.set_leaf_labels(sdg.fit_predict(predictions, *args, **kwargs))

        return self.leaf_labels

    def node_change_absolute(self, nodes1, nodes2):
        # First we obtain the medians for the nodes in question
        n1_predictions = np.mean(
            self.node_representation(nodes1, mode='mean'), axis=0)
        n2_predictions = np.mean(
            self.node_representation(nodes2, mode='mean'), axis=0)
        difference = n2_predictions - n1_predictions

        # Then sort by difference and return
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.output_features)[feature_order]
        ordered_difference = difference[feature_order]

        return ordered_features, ordered_difference

    def node_change_log_fold(self, nodes1, nodes2):

        # First we obtain the medians for the nodes in question
        n1_means = np.mean(self.node_representation(
            nodes1, mode='mean'), axis=0)
        n2_means = np.mean(self.node_representation(
            nodes2, mode='mean'), axis=0)

        # We evaluate the ratio of median values
        # log_fold_change = np.log2(n2_medians/n1_medians)
        log_fold_change = np.log2(n2_means / n1_means)

        # Because we are working with a division and a log, we have to filter for
        # results that don't have division by zero issues

        degenerate_mask = np.isfinite(log_fold_change)
        non_degenerate_features = np.array(self.output_features)[
            degenerate_mask]
        non_degenerate_changes = log_fold_change[degenerate_mask]

        # Finally sort and return
        feature_order = np.argsort(non_degenerate_changes)
        ordered_features = non_degenerate_features[feature_order]
        ordered_difference = non_degenerate_changes[feature_order]

        return ordered_features, ordered_difference

    def node_change_logistic(self, nodes1, nodes2):

        from sklearn.linear_model import LogisticRegression

        n1_counts = self.mean_matrix(nodes1)
        n2_counts = self.mean_matrix(nodes2)

        combined = np.concatenate([n1_counts, n2_counts], axis=0)

        scaled = sklearn.preprocessing.scale(combined)

        labels = np.zeros(n1_counts.shape[0] + n1_counts.shape[0])

        labels[n2_counts.shape[0]:] = 1
        #
        # print(f"Logistic debug:{n1_counts.shape},{n2_counts.shape},{combined.shape},{labels.shape}")

        model = LogisticRegression().fit(combined, labels)

        feature_sort = np.argsort(model.coef_[0, :])

        ordered_features = np.array(self.output_features)[feature_sort]
        ordered_coefficients = model.coef_[0, :][feature_sort]

        return ordered_features, ordered_coefficients

    def interpret_splits(self, override=False, mode='partial', metric='cosine', pca=100, relatives=True, resolution=1, k=10, depth=6, **kwargs):

        if pca > len(self.output_features):
            print(
                "WARNING, PCA DIMENSION GREATER THAN FEATURE DIMENSION, PICKING MINIMUM")
            pca = len(self.output_features)

        nodes = np.array(self.nodes(root=True, depth=depth))

        stem_mask = np.array([n.level != 0 for n in nodes])
        root_mask = np.logical_not(stem_mask)

        labels = np.zeros(len(nodes)).astype(dtype=int)

        if relatives:

            own_representation = self.node_representation(
                nodes[stem_mask], mode=mode, pca=pca)
            sister_representation = self.node_representation(
                [n.sister() for n in nodes[stem_mask]], mode=mode, pca=pca)

            knn = double_fast_knn(own_representation,
                                  sister_representation, k=k, metric=metric, **kwargs)

        else:
            representation = self.node_representation(
                nodes[stem_mask], mode=mode, pca=pca)

            knn = fast_knn(representation, k=k, metric=metric, **kwargs)

        labels[stem_mask] = 1 + hacked_louvain(knn, resolution=resolution)

        for node, label in zip(nodes, labels):
            node.set_split_cluster(label)

        cluster_set = set(labels)
        clusters = []
        for cluster in cluster_set:
            split_index = np.arange(len(labels))[labels == cluster]
            clusters.append(NodeCluster(
                self, [nodes[i] for i in split_index], cluster))

        self.split_clusters = clusters
        self.factors = self.split_clusters

        return labels

    def external_split_labels(self, nodes, labels, roots=False):

        # Use this method if you already know what some the nodes are labeled.

        cluster_set = set(labels)
        clusters = []

        for node, label in zip(nodes, labels):
            node.set_split_cluster(label)
            # node.split_cluster = label

        if not roots:
            for node in self.roots():
                node.set_split_cluster(0)
            clusters.append(NodeCluster(self, self.roots(), 0))

        for cluster in cluster_set:
            split_index = np.arange(len(labels))[labels == cluster]
            clusters.append(NodeCluster(
                self, [nodes[i] for i in split_index], cluster))

        self.split_clusters = clusters

    def create_root_cluster(self):

        roots = [t.root for t in self.trees]

        for node in roots:
            node.set_split_cluster(0)

        self.split_clusters = [NodeCluster(self, roots, 0), ]

    def reset_sample_clusters(self):
        try:
            for i in range(len(self.sample_clusters)):
                self.remove_output_feature(f'sample_cluster_{i}')
            del self.sample_clusters
            del self.sample_cluster_encoding
            del self.sample_labels
        except Exception:
            print("No sample clusters")

    def reset_split_clusters(self):
        try:
            del self.split_clusters
            for node in self.nodes():
                node.child_clusters = ([], [])
                if hasattr(node, 'split_cluster'):
                    del node.split_cluster
        except Exception:
            print("No split clusters")

    def reset_leaf_clusters(self):
        try:
            del self.leaf_clusters
            del self.leaf_labels
            for node in self.nodes():
                if hasattr(node, 'leaf_cluster'):
                    del node.leaf_cluster
        except Exception:
            print("No leaf clusters")

    def reset_clusters(self):

        self.reset_sample_clusters()
        self.reset_split_clusters()
        self.reset_leaf_clusters()

########################################################################
########################################################################

# PLOTTING METHODS

########################################################################
########################################################################

    def plot_sample_clusters(self, colorize=True, label=True):

        if not hasattr(self, 'sample_clusters'):
            print("Warning, sample clusters not detected")
            return None

        coordinates = self.coordinates(no_plot=True)

        cluster_coordiantes = np.zeros((len(self.sample_clusters), 2))

        for i, cluster in enumerate(self.sample_clusters):
            cluster_sample_mask = self.sample_labels == cluster.id
            mean_coordinates = np.mean(
                coordinates[cluster_sample_mask], axis=0)
            cluster_coordiantes[i] = mean_coordinates

        combined_coordinates = np.zeros(
            (self.output.shape[0] + len(self.sample_clusters), 2))

        combined_coordinates[0:self.output.shape[0]] = coordinates

        combined_coordinates[self.output.shape[0]:] = cluster_coordiantes

        highlight = np.ones(combined_coordinates.shape[0]) * 10
        highlight[len(self.sample_labels):] = [len(cluster.samples)
                                               for cluster in self.sample_clusters]

        combined_labels = np.zeros(
            self.output.shape[0] + len(self.sample_clusters))
        if colorize:
            combined_labels[0:len(self.sample_labels)] = self.sample_labels
            combined_labels[len(self.sample_labels):] = [
                cluster.id for cluster in self.sample_clusters]

        cluster_names = [cluster.name() for cluster in self.sample_clusters]
        cluster_coordiantes = combined_coordinates[len(self.sample_labels):]

        if label:
            f = plt.figure(figsize=(10, 10))
            plt.title("Sample Coordinates")
            plt.scatter(combined_coordinates[:, 0], combined_coordinates[:,
                                                                         1], s=highlight, c=combined_labels, cmap='rainbow')
            for cluster, coordinates in zip(cluster_names, cluster_coordiantes):
                plt.text(*coordinates, cluster, verticalalignment='center',
                         horizontalalignment='center')
        else:
            f = plt.figure(figsize=(20, 20))
            plt.title("Sample Coordinates")
            plt.scatter(combined_coordinates[:len(self.samples), 0], combined_coordinates[:len(
                self.samples), 1], s=10, c=combined_labels[:len(self.samples)], cmap='rainbow')
            plt.savefig("./tmp.delete.png", dpi=300)
        return f

    def plot_split_clusters(self, colorize=True):

        if not hasattr(self, 'split_clusters'):
            print("Warning, split clusters not detected")
            return None

        coordinates = self.coordinates(no_plot=True)

        cluster_coordinates = np.zeros((len(self.split_clusters), 2))

        for i, cluster in enumerate(self.split_clusters):
            cluster_coordinates[i] = cluster.coordinates(
                coordinates=coordinates)

        combined_coordinates = np.zeros(
            (self.output.shape[0] + len(self.split_clusters), 2))

        combined_coordinates[0:self.output.shape[0]] = coordinates

        combined_coordinates[self.output.shape[0]:] = cluster_coordinates

        highlight = np.ones(combined_coordinates.shape[0])
        highlight[self.output.shape[0]:] = [
            len(cluster.nodes) for cluster in self.split_clusters]

        combined_labels = np.zeros(
            self.output.shape[0] + len(self.split_clusters))
        if colorize:
            combined_labels[self.output.shape[0]:] = [
                cluster.id for cluster in self.split_clusters]

        cluster_names = [cluster.name() for cluster in self.split_clusters]
        cluster_coordiantes = combined_coordinates[-1 *
                                                   len(self.split_clusters):]

        f = plt.figure(figsize=(5, 5))
        plt.title("TSNE-Transformed Sample Coordinates")
        plt.scatter(combined_coordinates[:, 0], combined_coordinates[:,
                                                                     1], s=highlight, c=combined_labels, cmap='rainbow')
        for cluster, coordinates in zip(cluster_names, cluster_coordiantes):
            plt.text(*coordinates, cluster, verticalalignment='center',
                     horizontalalignment='center')
        plt.show()

        return f

    def plot_representation(self, representation, labels=None, metric='cos', pca=False):

        if metric is not None:
            # image = reduction[split_order].T[split_order].T
            agg_f = dendrogram(linkage(
                representation, metric=metric, method='average'), no_plot=True)['leaves']
            agg_s = dendrogram(linkage(
                representation.T, metric=metric, method='average'), no_plot=True)['leaves']
            image = representation[agg_f].T[agg_s].T
        else:
            try:
                agg_f = dendrogram(linkage(
                    representation.T, metric='cosine', method='average'), no_plot=True)['leaves']
            except Exception:
                agg_f = dendrogram(linkage(
                    representation.T, metric='cityblock', method='average'), no_plot=True)['leaves']
            try:
                agg_s = dendrogram(linkage(
                    representation, metric='cosine', method='average'), no_plot=True)['leaves']
            except Exception:
                agg_s = dendrogram(linkage(
                    representation, metric='cityblock', method='average'), no_plot=True)['leaves']

            image = representation[agg_s].T[agg_f].T
            image = representation[agg_s].T[agg_f].T

        plt.figure(figsize=(10, 10))
        plt.imshow(image, aspect='auto', cmap='bwr')
        plt.show()

        if labels is not None:

            split_order = np.argsort(labels)

            image = representation[split_order].T[agg_f].T
            plt.figure(figsize=(10, 10))
            plt.imshow(image, aspect='auto', cmap='bwr')
            plt.show()

    def sample_cluster_feature_matrix(self, features=None):
        if features is None:
            features = self.output_features
        coordinates = np.zeros((len(self.sample_clusters), len(features)))
        for i, sample_cluster in enumerate(self.sample_clusters):
            for j, feature in enumerate(features):
                # coordinates[i,j] = sample_cluster.feature_median(feature)
                coordinates[i, j] = sample_cluster.feature_mean(feature)
        return coordinates

    def factor_partial_matrix(self, features=None):
        if features is None:
            coordinates = np.zeros(
                (len(self.split_clusters), len(self.output_features)))
            for i, split_cluster in enumerate(self.split_clusters):
                coordinates[i] = np.mean(
                    self.partial_matrix(split_cluster.nodes), axis=1)
        else:
            coordinates = np.zeros((len(self.split_clusters), len(features)))
            for i, split_cluster in enumerate(self.split_clusters):
                for j, feature in enumerate(features):
                    coordinates[i, j] = split_cluster.feature_partial(
                        feature)
        return coordinates

    def factor_feature_matrix(self, features=None):
        if features is None:
            coordinates = np.zeros(
                (len(self.split_clusters), len(self.output_features)))
            for i, split_cluster in enumerate(self.split_clusters):
                coordinates[i] = np.mean(
                    self.mean_additive_matrix(split_cluster.nodes), axis=1)
        else:
            coordinates = np.zeros((len(self.split_clusters), len(features)))
            for i, split_cluster in enumerate(self.split_clusters):
                for j, feature in enumerate(features):
                    coordinates[i, j] = split_cluster.feature_mean_additive(
                        feature)
        return coordinates

    def factor_mean_matrix(self, features=None):
        if features is None:
            coordinates = np.zeros(
                (len(self.split_clusters), len(self.output_features)))
            for i, split_cluster in enumerate(self.split_clusters):
                coordinates[i] = np.mean(
                    self.mean_matrix(split_cluster.nodes), axis=0)
        else:
            coordinates = np.zeros((len(self.split_clusters), len(features)))
            for i, split_cluster in enumerate(self.split_clusters):
                for j, feature in enumerate(features):
                    coordinates[i, j] = split_cluster.feature_mean(feature)
        return coordinates

    def tsne(self, no_plot=False, pca=100, override=False, **kwargs):
        if not hasattr(self, 'tsne_coordinates') or override:
            if pca:
                pca = np.min([pca, self.output.shape[0], self.output.shape[1]])
                self.tsne_coordinates = TSNE().fit_transform(
                    PCA(n_components=pca).fit_transform(self.output))
            else:
                self.tsne_coordinates = TSNE().fit_transform(self.output)

        if not no_plot:
            plt.figure()
            plt.title("TSNE-Transformed Sample Coordinates")
            plt.scatter(
                self.tsne_coordinates[:, 0], self.tsne_coordinates[:, 1], s=.1, **kwargs)
            plt.show()

        return self.tsne_coordinates

    def tsne_encoding(self, no_plot=False, override=False, pca=False, **kwargs):
        if not hasattr(self, 'tsne_coordinates') or override:
            if pca:
                self.tsne_coordinates = TSNE().fit_transform(
                    PCA(n_components=pca).fit_transform(self.node_sample_encoding(self.leaves())))
            else:
                self.tsne_coordinates = TSNE().fit_transform(
                    self.node_sample_encoding(self.leaves()))

        if not no_plot:
            plt.figure()
            plt.title("TSNE-Transformed Sample Coordinates")
            plt.scatter(
                self.tsne_coordinates[:, 0], self.tsne_coordinates[:, 1], s=.1, **kwargs)
            plt.show()

        return self.tsne_coordinates

    def pca(self, no_plot=False, override=False, **kwargs):
        if not hasattr(self, 'pca_coordinates') or override:
            self.pca_coordinates = PCA(
                n_components=2).fit_transform(self.output)

        if not no_plot:
            plt.figure()
            plt.title("PCA-Transformed Sample Coordinates")
            plt.scatter(
                self.pca_coordinates[:, 0], self.pca_coordinates[:, 1], s=.1, **kwargs)
            plt.show()

        return self.pca_coordinates

    def umap(self, no_plot=False, override=False, **kwargs):
        if not hasattr(self, 'umap_coordinates') or override:
            self.umap_coordinates = UMAP().fit_transform(self.output)

        if not no_plot:
            plt.figure()
            plt.title("UMAP-Transformed Sample Coordinates")
            plt.scatter(
                self.umap_coordinates[:, 0], self.umap_coordinates[:, 1], s=.1, **kwargs)
            plt.show()

        return self.umap_coordinates

    def umap_encoding(self, no_plot=False, override=False, **kwargs):
        if not hasattr(self, 'umap_coordinates') or override:
            self.umap_coordinates = UMAP().fit_transform(
                self.node_sample_encoding(self.leaves()))

        if not no_plot:
            plt.figure()
            plt.title("UMAP-Transformed Sample Coordinates")
            plt.scatter(
                self.umap_coordinates[:, 0], self.umap_coordinates[:, 1], s=.1, **kwargs)
            plt.show()

        return self.umap_coordinates

    def coordinates(self, type=None, scaled=True, **kwargs):

        if type is None:
            if hasattr(self, 'coordinate_type'):
                type = self.coordinate_type
            else:
                type = 'tsne'

        self.coordinate_type = type

        type_functions = {
            'tsne': self.tsne,
            'tsne_encoding': self.tsne_encoding,
            'pca': self.pca,
            'umap': self.umap,
            'umap_encoding': self.umap_encoding,
        }

        coordinates = type_functions[type](**kwargs)

        if scaled:
            coordinates = sklearn.preprocessing.scale(coordinates)

        return coordinates

    def plot_manifold(self, depth=3):

        f = self.plot_sample_clusters()

        def recursive_tree_plot(parent, children, figure, level=0):
            pc = self.split_clusters[parent].coordinates()
            vectors = []
            for child, sub_children in children:
                if child == len(self.split_clusters):
                    continue
                cc = self.split_clusters[child].coordinates()
                v = cc - pc
                plt.figure(figure.number)
                plt.arrow(pc[0], pc[1], v[0], v[1], length_includes_head=True)
                vectors.append((pc, v))
                figure, cv = recursive_tree_plot(
                    child, sub_children, figure, level=level + 1)
                vectors.extend(cv)
            return figure, vectors

        f, v = recursive_tree_plot(self.likely_tree[0], self.likely_tree[1], f)

        plt.savefig("./tmp.delete.png", dpi=300)

        return f, v

    def plot_braid_vectors(self):

        f = self.plot_sample_clusters(label=False)
        ax = f.add_axes([0, 0, 1, 1])

        for cluster in self.split_clusters:
            ax = cluster.plot_braid_vectors(ax=ax, scatter=False, show=False)

        plt.savefig("./tmp.delete.png", dpi=300)

        return f


########################################################################
########################################################################

# Consensus tree methods

########################################################################
########################################################################


    def split_cluster_transition_matrix(self, depth=3):

        nodes = np.array(self.nodes(depth=depth))
        labels = self.split_labels(depth=depth)
        clusters = [cluster.id for cluster in self.split_clusters]
        transitions = np.zeros((len(clusters) + 1, len(clusters) + 1))

        for cluster in clusters:
            mask = labels == cluster
            cluster_nodes = nodes[mask]
            for node in cluster_nodes:
                node_state = node.split_cluster
                for child in node.children:
                    if hasattr(child, 'split_cluster'):
                        child_state = child.split_cluster
                    else:
                        child_state = len(clusters)
                    transitions[node_state, child_state] += 1
                if len(node.children) == 0:
                    transitions[node_state, -1] += 1
                if node.parent is None:
                    transitions[-1, node_state] += 1

        self.split_cluster_transitions = transitions

        return transitions

    def directional_matrix(self):

        downstream_frequency = np.zeros(
            (len(self.split_clusters), len(self.split_clusters)), dtype=int)
        upstream_frequency = np.zeros(
            (len(self.split_clusters), len(self.split_clusters)), dtype=int)

        for cluster in self.split_clusters:

            children = cluster.children()

            for child in children:
                if hasattr(child, 'split_cluster'):
                    downstream_frequency[cluster.id, child.split_cluster] += 1

            ancestors = cluster.ancestors()

            for ancestor in ancestors:
                if hasattr(ancestor, 'split_cluster'):
                    upstream_frequency[cluster.id, ancestor.split_cluster] += 1

        return upstream_frequency, downstream_frequency

    def path_matrix(self, nodes=None):

        if nodes is None:
            nodes = self.nodes()

        mtx = np.zeros((len(self.split_clusters), len(nodes)))
        for node in nodes:
            if hasattr(node, 'split_cluster'):
                mtx[node.split_cluster, node.index] = True

        return mtx

    def partial_dependence(self):
        path_matrix = self.path_matrix()
        path_covariance = np.cov(path_matrix)
        precision = np.linalg.pinv(path_covariance)

        precision_normalization = np.sqrt(
            np.outer(np.diag(precision), np.diag(precision)))
        path_partials = precision / precision_normalization

        path_partials[np.isnan(path_partials)] = 0

        return path_partials

    def split_cluster_odds_ratios(self):

        cluster_populations = [len(c.nodes) for c in self.split_clusters]
        total_nodes = np.sum(cluster_populations)

        downstream_frequency = np.ones(
            (len(self.split_clusters), len(self.split_clusters)))
        nephew_frequency = np.ones(
            (len(self.split_clusters), len(self.split_clusters)))

        for cluster in self.split_clusters[1:]:

            children = cluster.children()
            nephews = cluster.sisters() + [s.nodes()
                                           for s in cluster.sisters()]

            for child in children:
                if hasattr(child, 'split_cluster'):
                    downstream_frequency[cluster.id, child.split_cluster] += 1

            for nephew in nephews:
                if hasattr(nephew, 'split_cluster'):
                    nephew_frequency[cluster.id, nephew.split_cluster] += 1

            downstream_frequency[cluster.id] /= len(cluster.nodes) + 1
            nephew_frequency[cluster.id] /= len(cluster.nodes) + 1

        odds_ratio = downstream_frequency / nephew_frequency

        return odds_ratio

    ###############
    # Here we have several alternative methods for constructing the consensus tree.
    ###############

    # Most of them depend on these two helper methods that belong only in this scope

    # The finite tree method takes a prototype, which is a list of lists.
    # Each element in the list corresponds to which elements consider this element their parent

    def finite_tree(cluster, prototype, available):
        print(cluster)
        print(prototype)
        children = []
        try:
            available.remove(cluster)
        except Exception:
            pass
        for child in prototype[cluster]:
            if child in available:
                available.remove(child)
                children.append(child)
        return [cluster, [Forest.finite_tree(child, prototype, available) for child in children]]

    def reverse_tree(tree):
        root = tree[0]
        sub_trees = tree[1]
        child_entries = {}
        for sub_tree in sub_trees:
            for child, path in Forest.reverse_tree(sub_tree).items():
                path.append(root)
                child_entries[child] = path
        child_entries[root] = []
        return child_entries

    # End helpers

    def most_likely_tree(self, depth=3, transitions=None):

        if transitions is None:
            transitions = self.split_cluster_transition_matrix(depth=depth)

        transitions[np.identity(transitions.shape[0]).astype(dtype=bool)] = 0

        clusters = [cluster.id for cluster in self.split_clusters]

        proto_tree = [[] for cluster in clusters]

        for cluster in clusters:
            parent = np.argmax(transitions[:-1, cluster])
            proto_tree[parent].append(cluster)

        print(f"Clusters:{clusters}")
        print(f"Prototype:{proto_tree}")
        print(f"Prototype length:{len(proto_tree)}")
        print(f"Transitions:{transitions}")
        print(f"Transition mtx shape: {transitions.shape}")

        tree = []
        # entry = np.argmax(transitions[-1])
        entry = 0

        tree = Forest.finite_tree(
            cluster=entry, prototype=proto_tree, available=clusters)
        rtree = Forest.reverse_tree(tree)

        self.likely_tree = tree
        self.reverse_likely_tree = rtree

        return tree

    def maximum_spanning_tree(self, depth=3, mode="transition_matrix", transitions=None):

        if mode == "transition_matrix":
            distances = self.split_cluster_transition_matrix(depth=depth)
            # np.diag(distances) = 0
            distances[:, -1] = 0
        elif mode == "odds_ratio":
            distance = 1. / self.split_cluster_odds_ratios()
        elif mode == "dependence":
            distance = self.partial_dependence()
        elif mode == "means":
            mean_matrix = self.split_cluster_mean_matrix()
            domain_matrix = self.split_cluster_domain_mean_matrix()
            distances = 1. - cdist(mean_matrix, domain_matrix, metric="cosine")
        elif mode == "samples":
            cluster_values = np.array([c.sample_scores()
                                       for c in self.split_clusters])
            parent_values = np.array([c.parent_scores()
                                      for c in self.split_clusters])

            normed_cv = cluster_values / np.sum(cluster_values, axis=0)
            normed_pv = parent_values / np.sum(parent_values, axis=0)
            distances = 1. - cdist(normed_cv, normed_pv, metric='cosine')

        else:
            raise Exception(f"Not a valid mode: {mode}")

        mst = np.array(scipy.sparse.csgraph.minimum_spanning_tree(
            distances * -1).todense()) * -1

        mst = np.maximum(mst, mst.T)

        clusters = set(range(len(self.split_clusters)))

        print("Max tree debug")
        print(distances)
        print(mst)

        def finite_tree(cluster, available):
            children = []
            try:
                available.remove(cluster)
            except Exception:
                pass
            for child in np.arange(mst.shape[0])[mst[cluster] > 0]:
                if child in available:
                    available.remove(child)
                    children.append(child)
            return [cluster, [finite_tree(child, available) for child in children]]

        tree = finite_tree(0, clusters)
        rtree = Forest.reverse_tree(tree)

        self.likely_tree = tree
        self.reverse_likely_tree = rtree

        return tree


#########################################################
# HTML Visualization methods
#########################################################


    def html_directory(self):

        location = Path(__file__).parent.absolute()
        location = str(location) + "/html/"

        return location

    def location(self):
        return str(Path(__file__).parent.absolute())

    def html_tree_summary(self, n=3, mode="ud", custom=None, labels=None, features=None, primary=True, cmap='viridis', secondary=True, figsize=(30, 30), output=None):

        # First we'd like to make sure we are operating from scratch in the html directory:

        if n > (self.output.shape[1] / 2):
            print("WARNING, PICKED N THAT IS TOO LARGE, SETTING LOWER")
            n = max(int(self.output.shape[1] / 2), 1)

        if output is None:
            location = self.location()
            html_location = self.html_directory()
            rmtree(html_location)
            makedirs(html_location)
        else:
            location = self.location()
            html_location = output

        for split_cluster in self.split_clusters:
            print(f"Summarizing:{split_cluster.name()}")
            split_cluster.html_cluster_summary(
                n=n, plot=False, output=html_location + str(split_cluster.id) + "/")

        copyfile(location + "/tree_template.html",
                 html_location + "tree_template.html")

        with open(html_location + "tree_template.html", 'a') as html_report:

            # Helper methods:

            # Find the leaves of a tree
            def leaves(tree):
                l = []
                for child in tree[1]:
                    l.extend(leaves(child))
                if len(l) < 1:
                    l.append(tree[0])
                return l

            # Find out how many levels are in this tree (eg its maximum depth)
            def levels(tree, level=0):
                l = []
                for child in tree[1]:
                    l.extend(levels(child, level=level + 1))
                l.append(level)
                return l

            # First we compute the width/height of the individual cells
            width = 1 / len(leaves(self.likely_tree))
            height = 1 / (max(levels(self.likely_tree)) + 1)

            # This function determines where to place everything, all coordinates are from 0 to 1, so are fractions of a canvas.
            # The coordinates are placed in a list we pass to the function, because I didn't want to think about how to avoid doing this
            def recursive_axis_coordinates(tree, child_coordinates, limits=[0, 0]):
                [x, y] = limits
                child_width = 0
                # First we go lower in recursive layer and find how many children we need to account for from this leaf
                for child in tree[1]:
                    cw = recursive_axis_coordinates(child, child_coordinates, [
                                                    x + child_width, y + (height)])
                    child_width += cw
                if child_width == 0:
                    child_width = width

                # We have to place the current leaf at the average position of all leaves below
                padding = (child_width - width) / 2
                coordinates = [x + padding + (width * .5), y + (height * .5)]

                child_coordinates.append([int(tree[0]), coordinates])

                return child_width

            # Here we actually call the recursive function

            coordinates = [[width, height], ]
            recursive_axis_coordinates(self.likely_tree, coordinates)
            coordinates[1:] = list(sorted(coordinates[1:], key=lambda x: x[0]))

            # Now we have to create an HTML-ish string in order to pass this information on to
            # the javascript without reading local files (siiigh)

            coordinate_json_string = jsn_dumps(coordinates)
            name_json_string = jsn_dumps(
                [c.name() for c in self.split_clusters])
            coordinate_html = f'<script> let treeCoordinates = {coordinate_json_string};</script>'
            name_html = f'<script> let clusterNames = {name_json_string};</script>'

            # Finally, we append to the template to pass on the information

            html_report.write(coordinate_html)
            html_report.write(name_html)

            # Next we want to calculate the connections between each node:

            # First we flatten the tree:

            def flatten_tree(tree):
                flat = []
                for child in tree[1]:
                    flat.extend(flatten_tree(child))
                flat.append([tree[0], [c[0] for c in tree[1]]])
                return flat

            flat_tree = flatten_tree(self.likely_tree)

            # Then we insert connections:

            if primary:

                # print(f"Coordinates:{coordinates}")
                # print(f"Flat tree:{flat_tree}")

                primary_connections = []

                for i, children in flat_tree:
                    x, y = coordinates[1:][i][1]
                    center_x = x + (width * .5)
                    center_y = y + (height * .5)
                    for ci in children:
                        cx, cy = coordinates[1:][ci][1]
                        child_center_x = cx + (width / 2)
                        child_center_y = cy + (height / 2)

                        # We would like to set the arrow thickness to be proportional to the mean population of the child
                        if ci < len(self.split_clusters):
                            cp = self.split_clusters[ci].mean_population()
                        else:
                            cp = 1
                        primary_connections.append([x, y, cx, cy, cp])
                        # primary_connections.append([center_x,center_y,child_center_x,child_center_y,cp])

                primary_connection_json = jsn_dumps(primary_connections)
                primary_connection_html = f'<script>let connections = {primary_connection_json};</script>'

                # Again, we append to the template to pass on the information

                html_report.write(primary_connection_html)

            elif secondary:

                # If we want to indicate secondary connections:
                secondary_connections = []

                for i in range(len(self.split_clusters)):
                    for j in range(len(self.split_clusters)):

                        # We scroll through every element in the split cluster transition
                        # matrix

                        # if self.split_cluster_transitions[i,j] > 0:
                        if self.dependence_scores[i, j] < 0:

                            # If the transitions are non-zero we obtain the coordinates

                            x, y = coordinates[1:][i][1]
                            center_x = x + (width * .5)
                            center_y = y - (height * .5)
                            cx, cy = coordinates[1:][j][1]
                            child_center_x = cx + (width / 2)
                            child_center_y = cy + (height / 2)

                            # Alternatively, plot a line with a weight equivalent to the partial correlation of split clusters:

                            cp = self.dependence_scores[i, j]
                            secondary_connections.append(
                                [center_x, center_y, child_center_x, child_center_y, cp])

                secondary_connection_json = jsn_dumps(secondary_connections)
                secondary_connection_html = f'<script> let connections = {secondary_connection_json}</script>'

                # Finally, we append to the template to pass on the information

                html_report.write(secondary_connection_html)

            else:
                raise Exception("Pick a connectivity!")

            # Now we need to loop over available clusters to place the cluster decorations into the template

            for cluster in self.split_clusters:
                cluster_summary_json = cluster.json_cluster_summary(n=n)
                cluster_summary_html = f'<script> summaries["cluster_{cluster.id}"] = {cluster_summary_json};</script>'
                html_report.write(cluster_summary_html)

        from subprocess import run
        try:
            run(["open", html_location + "tree_template.html"])
        except:
            print(
                f"Stored the html report at {html_location}, but could not open from command line")

    def split_cluster_leaves(self):
        def tree_leaves(tree):
            leaves = []
            for child in tree[1]:
                leaves.extend(tree_leaves(child))
            if len(tree[1]) < 1:
                leaves.append(tree[0])
            return leaves

        tree = self.likely_tree
        leaf_clusters = [self.split_clusters[i] for i in tree_leaves(tree)]

        return leaf_clusters

    def cluster_samples_by_split_clusters(self, override=False, *args, **kwargs):

        if hasattr(self, 'sample_clusters') and not override:
            print("Clustering has already been done")
            return self.sample_labels

        leaf_split_clusters = self.split_cluster_leaves()
        leaf_split_cluster_sample_scores = np.array(
            [c.sample_counts() for c in leaf_split_clusters])
        sample_labels = np.array([np.argmax(
            leaf_split_cluster_sample_scores[:, i]) for i in range(len(self.samples))])

        self.set_sample_labels(sample_labels)

        print([c.id for c in self.split_clusters])

        return self.sample_labels

    def factor_matrix(self):
        matrix = np.zeros((len(self.samples), len(self.split_clusters)))
        for i, cluster in enumerate(self.split_clusters):
            matrix[:, i] = cluster.sister_scores()
        return matrix

    def global_correlations(self, indices=None):

        if indices is None:
            indices = np.arange(self.output.shape[1])

        correlations = np.corrcoef(self.output.T[indices])

        return correlations


class TruthDictionary:

    def __init__(self, counts, header, samples=None):

        self.counts = counts
        self.header = header
        self.feature_dictionary = {}

        self.sample_dictionary = {}
        for i, feature in enumerate(header):
            self.feature_dictionary[feature.strip('""').strip("''")] = i
        if samples is None:
            samples = map(lambda x: str(x), range(counts.shape[0]))
        for i, sample in enumerate(samples):
            self.sample_dictionary[sample.strip("''").strip('""')] = i

    def look(self, sample, feature):
        #         print(feature)
        return(self.counts[self.sample_dictionary[sample], self.feature_dictionary[feature]])


if __name__ != "__main__":
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 300
