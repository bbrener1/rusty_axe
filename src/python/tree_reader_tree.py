from tree_reader_node import Node

import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import copy,deepcopy

mpl.rcParams['figure.dpi'] = 100

class Tree:

    def __init__(self, tree_json, forest):
        self.root = Node(tree_json, self, forest, cache=forest.cache)
        self.forest = forest

    def nodes(self, root=True):
        nodes = []
        nodes.extend(self.root.nodes())
        if root:
            nodes.append(self.root)
        return nodes

    def leaves(self, depth=None):
        leaves = self.root.leaves(depth=depth)
        if len(leaves) < 1:
            leaves.append(self.root)
        return leaves

    def stems(self):
        stems = self.root.stems()
        return stems

    def level(self, target):
        level_nodes = []
        for node in self.nodes():
            if node.level == target:
                level_nodes.append(node)
        return level_nodes

    def descend(self, level):
        return self.root.descend(level)

    def seek(self, directions):
        if len(directions) > 0:
            self.children[directions[0]].seek(directions[1:])
        else:
            return self

    def trim(self,limit):
        for child in self.root.children:
            child.trim(limit)

    def derive_samples(self,samples):
        root_copy = self.root.derive_samples(samples)
        self_copy = self.derived_copy()
        self_copy.root = root_copy
        for node in self_copy.root.nodes():
            node.tree = self_copy
        self_copy.root.tree = self_copy
        return self_copy

    def derived_copy(self):
        self_copy = copy(self)
        self_copy.root = None
        self_copy.forest = None
        return self_copy

    def feature_levels(self):
        return self.root.feature_levels()

    def plotting_representation(self, width=10, height=10):
        coordinates = []
        connectivities = []
        bars = []
        levels = self.root.nodes_by_level()
        jump = height / len(levels)
        for i, level in enumerate(levels):
            level_samples = sum([node.pop() for node in level])
            next_level_samples = 0
            if i < (len(levels) - 1):
                next_level_samples = sum([node.pop()
                                          for node in levels[i + 1]])
            consumed_width = 0
            next_consumed_width = 0
            for j, node in enumerate(level):
                sample_weight = float(node.pop()) / float(level_samples)
                half_width = (width * sample_weight) / 2
                center = consumed_width + half_width
                consumed_width = consumed_width + (half_width * 2)
                coordinates.append((i * jump, center))
                if i < (len(levels) - 1):
                    for child in node.children:
                        child_sample_weight = float(
                            child.pop()) / float(next_level_samples)
                        child_half_width = (width * child_sample_weight) / 2
                        child_center = next_consumed_width + child_half_width
                        next_consumed_width = next_consumed_width + \
                            (child_half_width * 2)
                        connectivities.append(
                            ([i * jump, (i + 1) * jump], [center, child_center]))
        coordinates = np.array(coordinates)
        plt.figure()
        plt.scatter(coordinates[:, 0], coordinates[:, 1], s=1)
        for connection in connectivities:
            plt.plot(connection[0], connection[1])
        plt.show()

        # return coordinates,connectivities

    def recursive_plotting_repesentation(self, axes, height=None, height_step=None, representation=None, limits=None):
        if limits is None:
            limits = axes.get_xlim()
        current_position = limits[0]
        width = float(limits[1] - limits[0])
        center = (limits[1] + limits[0]) / 2
        if representation is None:
            representation = self.root.plotting_representation()
            print(representation)
        if height_step is None or height is None:
            depth = self.root.depth()
            height_limits = axes.get_ylim()
            height = height_limits[1]
            height_step = -1 * (height_limits[1] - height_limits[0]) / depth
        # print(representation)
        for i, current_representation in enumerate(representation):
            width_proportion = current_representation[0]
            children = current_representation[1]
            node_start = current_position
            node_width = width_proportion * width
            padding = node_width * .05
            node_width = node_width - padding
            node_center = (node_width / 2) + current_position
            node_height = height + height_step
            node_end = (node_width) + current_position
            current_position = node_end + padding

            color = ['r', 'b'][(i % 2)]

            axes.plot([center, node_center], [height, node_height], c=color)
            # axes.plot([node_center],[node_height])
            axes.plot([node_start, node_end], [
                      node_height, node_height], c=color)

            self.recursive_plotting_repesentation(
                axes, height=node_height, height_step=height_step, representation=children, limits=(node_start, node_end))

    def plot(self):
        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(111)
        self.recursive_plotting_repesentation(ax)
        fig.show()

    def tree_movie_frame(self, location, level=0, sorted=True, previous_frame=None, split_lines=True):
        descent_nodes = self.descend(level)
        total_samples = sum([node.pop() for node in descent_nodes])
        heatmap = np.zeros((total_samples, len(self.forest.output_features)))
        node_splits = []
        running_samples = 0
        for node in descent_nodes:
            if sorted:
                node_counts = node.sorted_node_counts()
            else:
                node_counts = node.node_counts()
            node_samples = node_counts.shape[0]
            heatmap[running_samples:running_samples +
                    node_samples] = node_counts
            running_samples += node_samples
            node_splits.append(running_samples)
        plt.figure(figsize=(10, 10))
        if previous_frame is None:
            plt.imshow(heatmap, aspect='auto')
        else:
            plt.imshow(previous_frame, aspect='auto')
        if split_lines:
            for split in node_splits[:-1]:
                plt.plot([0, len(self.forest.output_features) - 1],
                         [split, split], color='w')
        plt.savefig(location)
        return heatmap

    def tree_movie(self, location):
        max_depth = max([leaf.level for leaf in self.leaves()])
        previous_frame = None
        for i in range(max_depth):
            self.tree_movie_frame(location + "." + str(i) + ".a.png",
                                  level=i, sorted=False, previous_frame=previous_frame)
            previous_frame = self.tree_movie_frame(
                location + "." + str(i) + ".b.png", level=i, sorted=True)
        self.tree_movie_frame(location + "." + str(i + 1) +
                              ".b.png", level=i, sorted=True, split_lines=False)

    def summary(self, verbose=True):
        nodes = len(self.nodes)
        leaves = len(self.leaves)
        if verbose:
            print("Nodes: {}".format(nodes))
            print("Leaves: {}".format(leaves))

    def aborting_sample_descent(self, sample):
        return self.root.aborting_sample_descent(sample)

    def plot_leaf_counts(self):
        leaves = self.leaves()
        total_samples = sum([x.pop() for x in leaves])
        heatmap = np.zeros((total_samples, len(self.forest.output_features)))
        running_samples = 0
        for leaf in leaves:
            leaf_counts = leaf.node_counts()
            leaf_samples = leaf_counts.shape[0]
            heatmap[running_samples:running_samples +
                    leaf_samples] = leaf_counts
            running_samples += leaf_samples

        ordering = dendrogram(linkage(heatmap.T), no_plot=True)['leaves']
        heatmap = heatmap.T[ordering].T
        plt.figure()
        im = plt.imshow(heatmap, aspect='auto')
        plt.colorbar()
        plt.show()

        return heatmap
    # def cluster_distances(self):
    #     for leaf in self.leaves():
