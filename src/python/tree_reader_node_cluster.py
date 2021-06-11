from tree_reader_utils import js_wrap, generate_feature_value_html, generate_cross_reference_table, weighted_correlation
import os

import numpy as np

from matplotlib.colors import DivergingNorm


import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.rcParams['figure.dpi'] = 10

DPI_SET = 100

class NodeCluster:

    def __init__(self, forest, nodes, id):
        self.id = id
        self.nodes = nodes
        self.forest = forest

    def name(self):
        if hasattr(self, 'stored_name'):
            return self.stored_name
        else:
            return str(self.id)

    def set_name(self, name):
        self.stored_name = name

################################################################################
# Basic manipulation methods (Nodes/Representations etc)
################################################################################

    def encoding(self):
        return self.forest.node_sample_encoding(self.nodes)

    def node_mask(self):
        mask = np.zeros(len(self.forest.nodes()), dtype=bool)
        indices = [n.index for n in self.nodes]
        mask[indices] = True
        return mask

    def parent_mask(self):
        mask = np.zeros(len(self.forest.nodes()), dtype=bool)
        parents = [p.index for p in self.parents()]
        mask[parents] = True
        return mask

    def sisters(self):
        return [n.sister() for n in self.nodes if n.sister() is not None]

    def children(self):

        return [c for n in self.nodes for c in n.nodes()]

    def parents(self):

        return [n.parent for n in self.nodes if n.parent is not None]

    def ancestors(self):

        return [a for n in self.nodes for a in n.ancestors()]
    #
    # def weighted_feature_predictions(self):
    #     return self.forest.weighted_node_vector_prediction(self.nodes)


################################################################################
# Consensus tree methods. Kinda weird/hacky. Need to rethink
################################################################################

    def parent_cluster(self):
        try:
            return self.forest.split_clusters[self.forest.reverse_likely_tree[self.id][0]]
        except:
            return self

    def child_clusters(self):

        def traverse(tree):
            children = []
            if tree[0] == self.id:
                children.extend([c[0] for c in tree[1]])
            else:
                for child in tree[1]:
                    children.extend(traverse(child))
            return children

        indices = traverse(self.forest.likely_tree)
        print(indices)
        return [self.forest.split_clusters[i] for i in indices]

    def sibling_clusters(self):

        def traverse(tree):
            siblings = []
            if self.id in [c[0] for c in tree[1]]:
                siblings.extend([c[0] for c in tree[1]])
                own_index = siblings.index(self.id)
                siblings.pop(own_index)
            else:
                for child in tree[1]:
                    siblings.extend(traverse(child))
            return siblings

        indices = traverse(self.forest.likely_tree)
        print(indices)
        return [self.forest.split_clusters[i] for i in indices]


##############################################################################
# Feature change methods (eg changes relative to related nodes)
##############################################################################

    def changed_absolute_root(self):
        roots = self.forest.nodes(root=True, depth=0)
        ordered_features, ordered_difference = self.forest.node_change_absolute(
            roots, self.nodes)
        return ordered_features, ordered_difference

    def changed_log_root(self):
        roots = self.forest.nodes(root=True, depth=0)
        ordered_features, ordered_difference = self.forest.node_change_log_fold(
            roots, self.nodes)
        return ordered_features, ordered_difference

    def changed_absolute(self):
        parents = [n.parent for n in self.nodes if n.parent is not None]
        ordered_features, ordered_difference = self.forest.node_change_absolute(
            parents, self.nodes)
        return ordered_features, ordered_difference

    def changed_log_fold(self):
        parents = [n.parent for n in self.nodes if n.parent is not None]
        ordered_features, ordered_difference = self.forest.node_change_log_fold(
            parents, self.nodes)
        return ordered_features, ordered_difference

    def changed_absolute_sister(self):
        sisters = [n.sister() for n in self.nodes if n.sister() is not None]
        ordered_features, ordered_difference = self.forest.node_change_absolute(
            sisters, self.nodes)
        return ordered_features, ordered_difference

    def changed_log_sister(self):
        sisters = [n.sister() for n in self.nodes if n.sister() is not None]
        ordered_features, ordered_difference = self.forest.node_change_log_fold(
            sisters, self.nodes)
        return ordered_features, ordered_difference

    def ranked_additive(self):
        additive = self.forest.node_representation(self.nodes, mode='additive')
        mean_additive = np.mean(additive, axis=0)
        sort = np.argsort(mean_additive)
        return self.forest.output_features[sort], mean_additive[sort]

    def log_sister(self, plot=True):
        sisters = [n.sister() for n in self.nodes]
        ordered_features, ordered_difference = self.forest.node_change_log_fold(
            sisters, self.nodes)
        return ordered_features, ordered_difference

    def logistic_sister(self, n=50, plot=True):
        sisters = [n.sister() for n in self.nodes]
        ordered_features, ordered_difference = self.forest.node_change_logistic(
            sisters, self.nodes)
        return ordered_features, ordered_difference


    def coordinates(self, coordinates=None):

        if coordinates is None:
            coordinates = self.forest.coordinates(no_plot=True)

        sample_scores = self.sample_scores()
        sample_scores = np.power(sample_scores, 2)
        mean_coordinates = np.dot(
            sample_scores, coordinates) / np.sum(sample_scores)

        return mean_coordinates

    def regression(self):
        from sklearn.linear_model import LinearRegression

        scores = self.sister_scores()
        weights = np.abs(scores)
        top_features = self.top_split_features()
        selection = self.forest.output.T[top_features].T

        factor_model = LinearRegression().fit(
            selection, scores.reshape(-1, 1), sample_weight=weights)
        output_model = LinearRegression().fit(
            selection, self.forest.output, sample_weight=weights)

        return top_features, factor_model, output_model

    def error_ratio(self):

        # We would like to weigh the observed error by by the total of the cluster and its sisters, then by samples in the cluster only
        # The ratio should give us an idea of how much of the variance is explained by the cluster split.

        sample_matrix = self.forest.output

        self_scores = self.sample_scores()

        self_mean = np.mean(self.forest.mean_matrix(self.nodes), axis=0)
        parent_mean = np.mean(
            self.forest.mean_matrix(self.parents()), axis=0)

        self_error = np.dot(
            np.power(sample_matrix - self_mean, 2).T, self_scores)
        parent_error = np.dot(
            np.power(sample_matrix - parent_mean, 2).T, self_scores)

        print(
            f"Error: P:{self_error},A:{parent_error}")

        error_ratio = (self_error + 1) / (parent_error + 1)

        print(f"Error ratio: {error_ratio}")

        ratio_sort = np.argsort(error_ratio)

        sorted_features = self.forest.output_features[ratio_sort]
        sorted_ratios = error_ratio[ratio_sort]

        return sorted_features, sorted_ratios

    # def raw_error(self):
    #
    #     # We want to figure out how much of the error in a given feature is explained by the nodes in this cluster
    #     # We want the overall ratio of error in the parents vs the error in the nodes of this cluster
    #
    #     parent_total_error = np.zeros(len(self.forest.output_features))
    #     sister_total_error = np.zeros(len(self.forest.output_features))
    #     self_total_error = np.zeros(len(self.forest.output_features))
    #
    #     for node in self.nodes:
    #         if node.parent is not None:
    #             self_total_error += node.squared_residual_sum()
    #             sister_total_error += node.sister().squared_residual_sum()
    #             parent_total_error += node.parent.squared_residual_sum()
    #
    #     return self_total_error,sister_total_error,parent_total_error

    def strict_error_ratio(self):

        self_total_error,parent_total_error = self.raw_error()

        self_total_error += 1
        parent_total_error += 1

        error_ratio = self_total_error / \
            parent_total_error

        ratio_sort = np.argsort(error_ratio)

        sorted_features = self.forest.output_features[ratio_sort]
        sorted_ratios = error_ratio[ratio_sort]

        return sorted_features, sorted_ratios

    def strict_fraction_unexplained(self):
        self_total_error,parent_total_error = self.raw_error()

        return (np.sum(self_total_error) + 1)  / (np.sum(parent_total_error) + 1)

    def raw_error(self):
        self_total_error = np.zeros(len(self.forest.output_features))
        parent_total_error = np.zeros(len(self.forest.output_features))

        for node in self.nodes:
            if node.parent is not None:
                self_error,parent_error = node.squared_residual_doublet()
                self_total_error += self_error
                parent_total_error += parent_error

        return self_total_error,parent_total_error


    def top_split_features(self, n=10):
        from sklearn.linear_model import LinearRegression

        split_features = list(
            set([n.feature() for n in self.nodes if n.feature() is not None]))
        selection = self.forest.output.T[split_features].T
        factor_scores = self.sister_scores()
        model = LinearRegression().fit(selection, factor_scores,
                                       sample_weight=np.abs(factor_scores))
        top_features = [split_features[i]
                        for i in np.argsort(np.abs(model.coef_))]
        return top_features[-n:]

################################################################################
# Mean/summary methods (describe cluster contents)
################################################################################

    def feature_mean(self, feature):
        return np.mean(self.forest.nodes_mean_predict_feature(self.nodes, feature))

    def feature_additive(self, feature):
        return np.mean([n.feature_additive(feature) for n in self.nodes])

    def feature_mean_additive(self, feature):
        return np.mean(self.forest.nodes_mean_additive_predict_feature(self.nodes, feature))

    def feature_means(self, features):
        return np.array([self.feature_mean(feature) for feature in features])

    def feature_additives(self, features):
        return np.array([self.feature_additive(feature) for feature in features])

    def feature_mean_additives(self, features):
        return np.array([self.feature_mean_additive(feature) for feature in features])

    def mean_level(self):
        return np.mean([n.level for n in self.nodes])

    def mean_population(self):
        return np.mean([n.pop() for n in self.nodes])

    def important_features(self,n=10,method='mean'):

        if method == 'mean':
            features, values = self.changed_absolute_sister()
            important_features = list(
                features[:n]) + list(features[-n:])
            important_values = list(values[:n]) + list(values[-n:])
        elif method == "error" or method == "cod":
            features, values = self.error_ratio()
            cod = 1 - error_ratio
            important_features = list(error_features[:n * 2])
            important_values = list(cod[:n * 2])
        else:
            raise Exception(f"Method not recognized: {method}")

        important_indices = [
            self.forest.truth_dictionary.feature_dictionary[f] for f in important_features]

        return important_features,important_values,important_indices

    def local_correlations(self, indices=None):

        if indices is None:
            indices = np.arange(self.forest.output.shape[1])

        weights = self.sample_counts()
        correlations = weighted_correlation(self.forest.output.T[indices],weights=weights)

        return correlations


    def most_local_correlations(self, n=10,method='mean'):

        _,_,important_indices = self.important_features(n=n,method=method)

        global_correlations = self.forest.global_correlations(indices=important_indices)
        local_correlations = self.local_correlations(indices=important_indices)

        delta = local_correlations - global_correlations

        ranks = np.argsort(np.abs(delta.flatten()))

        tiled_indices = np.tile(
            important_indices, ((delta.shape[0]), 1))

        ranked = list(zip(tiled_indices.flatten()[
            ranks], tiled_indices.T.flatten()[ranks]))

        return ranked[-n:]

##############################################################################
# Sample membership methods
##############################################################################

    def sample_scores(self):
        cluster_encoding = self.encoding()
        return np.sum(cluster_encoding, axis=1) / (cluster_encoding.shape[1] + 1)

    def parent_scores(self):
        if len(self.parents()) > 0:
            parent_encoding = self.forest.node_representation(self.parents(),mode='sample')
        else:
            parent_encoding = self.forest.node_representation(self.nodes,mode='sample')
        return np.sum(parent_encoding, axis=0) / (parent_encoding.shape[0] + 1)

    def sample_counts(self):
        encoding = self.encoding()
        return np.sum(encoding, axis=1)

    def sister_scores(self):
        own = self.nodes
        sisters = self.sisters()
        own_encoding = self.forest.node_sample_encoding(own).astype(dtype=int)
        sister_encoding = self.forest.node_sample_encoding(
            sisters).astype(dtype=int)
        scores = (np.sum(own_encoding, axis=1) + (-1 *
                                                  np.sum(sister_encoding, axis=1))) / own_encoding.shape[1]
        return scores

    def log_sister_scores(self,prior=1):
        own = self.nodes
        sisters = self.sisters()
        own_encoding = self.forest.node_sample_encoding(own).astype(dtype=int)
        sister_encoding = self.forest.node_sample_encoding(
            sisters).astype(dtype=int)
        ratio = (np.sum(own_encoding,axis=1) + prior) / (np.sum(sister_encoding,axis=1) + prior)

        return np.log(ratio)

    def predict_sister_scores(self, node_sample_encoding):
        own_nodes = self.nodes
        own_mask = np.zeros(node_sample_encoding.shape[0], dtype=bool)
        own_mask[[n.index for n in own_nodes]] = True

        sisters = self.sisters()
        sister_mask = np.zeros(node_sample_encoding.shape[0], dtype=bool)
        sister_mask[[s.index for s in sisters]] = True

        own_encoding = node_sample_encoding[own_mask]
        sister_encoding = node_sample_encoding[sister_mask]

        scores = (np.sum(own_encoding, axis=0) + (-1 *
                                                  np.sum(sister_encoding, axis=0))) / own_encoding.shape[0]

        return scores


    def predict_log_sister_scores(self, node_sample_encoding,prior=1):
        own_nodes = self.nodes
        own_mask = np.zeros(node_sample_encoding.shape[0], dtype=bool)
        own_mask[[n.index for n in own_nodes]] = True

        sisters = self.sisters()
        sister_mask = np.zeros(node_sample_encoding.shape[0], dtype=bool)
        sister_mask[[s.index for s in sisters]] = True

        own_encoding = node_sample_encoding[own_mask]
        sister_encoding = node_sample_encoding[sister_mask]

        ratio = (np.sum(own_encoding,axis=1) + prior) / (np.sum(sister_encoding,axis=1) + prior)

        return np.log(ratio)


##############################################################################
# Html methods
##############################################################################

# Methods here are used to generate HTML summaries of the cluster

    def html_directory(self):
        location = self.forest.html_directory() + str(self.id) + "/"
        if not os.path.exists(location):
            os.makedirs(location)
        return location

    def json_cluster_summary(self, n=20, features=None):

        # Summarizes the cluster in a json format, primarily for use in html summary documents
        # then returns the object wrapped in a

        from json import dumps as jsn_dumps

        attributes = {}

        error_features, error_ratio = self.error_ratio()
        coefficient_of_determination = 1 - error_ratio

        changed_vs_parent, fold_vs_parent = self.changed_log_fold()
        changed_vs_all, fold_vs_all = self.changed_log_root()
        changed_vs_sister, fold_vs_sister = self.changed_log_sister()

        changed_vs_parent, fold_vs_parent = self.changed_absolute()
        changed_vs_all, fold_vs_all = self.changed_absolute_root()
        changed_vs_sister, fold_vs_sister = self.changed_absolute_sister()

        # probability_enrichment = np.around(self.probability_enrichment(),3)
        probability_enrichment = np.around(self.odds_ratio(), 3)
        probability_enrichment = [(self.forest.split_clusters[i].name(), enrichment) for (
            i, enrichment) in enumerate(probability_enrichment)]

        local_cross_html,global_cross_html = self.html_cross_reference(n=n)

        attributes['clusterName'] = str(self.name())
        attributes['clusterId'] = int(self.id)
        attributes['errorUp'] = generate_feature_value_html(
            error_features[-n:], coefficient_of_determination[-n:], cmap='bwr')
        attributes['errorDown'] = generate_feature_value_html(
            error_features[:n], coefficient_of_determination[:n], cmap='bwr')
        attributes['parentUpregulatedHtml'] = generate_feature_value_html(
            reversed(changed_vs_parent[-n:]), reversed(fold_vs_parent[-n:]), cmap='bwr')
        attributes['parentDownregulatedHtml'] = generate_feature_value_html(
            reversed(changed_vs_parent[:n]), reversed(fold_vs_parent[:n]), cmap='bwr')
        attributes['sisterUpregulatedHtml'] = generate_feature_value_html(
            reversed(changed_vs_sister[-n:]), reversed(fold_vs_sister[-n:]), cmap='bwr')
        attributes['sisterDownregulatedHtml'] = generate_feature_value_html(
            reversed(changed_vs_sister[:n]), reversed(fold_vs_sister[:n]), cmap='bwr')
        attributes['absoluteUpregulatedHtml'] = generate_feature_value_html(
            reversed(changed_vs_all[-n:]), reversed(fold_vs_all[-n:]), cmap='bwr')
        attributes['absoluteDownregulatedHtml'] = generate_feature_value_html(
            reversed(changed_vs_all[:n]), reversed(fold_vs_all[:n]), cmap='bwr')
        attributes['probability_enrichment'] = probability_enrichment
        attributes['localCrossRef'] = local_cross_html
        attributes['globalCrossRef'] = global_cross_html
        # attributes['children'] = ", ".join(
        #     [c.name() for c in self.child_clusters()])
        # attributes['parent'] = self.parent_cluster().name()
        # attributes['siblings'] = ", ".join(
        #     [s.name() for s in self.sibling_clusters()])

        if features is not None:

            specified_feature_mask = [f in features for f in changed_features]
            specified_features = changed_features[specified_feature_mask]
            specified_feature_changes = change_fold[specified_feature_mask]

            attributes['specifiedHtml'] = generate_feature_value_html(
                specified_features, specified_feature_changes, cmap='bwr')

        return jsn_dumps(attributes)

    def top_local_table(self,n):
        # changed_vs_sister, fold_vs_sister = self.changed_absolute_sister()
        important_features, important_folds, important_indices = self.important_features(n)

        selected_local = self.local_correlations(indices=important_indices)

        selected_local = np.around(selected_local,decimals=3)

        return selected_local,important_features

    def top_global_table(self,n):
        changed_vs_sister, fold_vs_sister = self.changed_absolute_sister()
        important_features, important_folds, important_indices = self.important_features(n)
        selected_global = self.forest.global_correlations(indices=important_indices)

        selected_global = np.around(selected_global,decimals=3)

        return selected_global,important_features

    def top_local(self, n, no_plot=False):

        import matplotlib.patheffects as PathEffects

        selected_local,important_features = self.top_local_table(n)

        m = len(important_features)

        fig = plt.figure(figsize=(n, n))
        ax = fig.add_axes([0, 0, .8, 1])
        plt.title(f"Local Correlations in {self.name()}",fontsize=15)
        im = ax.imshow(selected_local, vmin=-1, vmax=1, cmap='bwr')
        for i in range(m):
            for j in range(m):
                text = ax.text(j, i, np.around(selected_local[i, j], 3),
                               ha="center", va="center", c='w', fontsize=7)
                text.set_path_effects(
                    [PathEffects.withStroke(linewidth=.5, foreground='black')])

        plt.xticks(np.arange(m), labels=important_features, rotation=45)
        plt.yticks(np.arange(m), labels=important_features, rotation=45)
        cb_ax = fig.add_axes([.85,.1,.1,.8])
        plt.colorbar(im,cax=cb_ax,label="Weighted Pearson Correlation")
        plt.tight_layout()
        if no_plot:
            return fig
        else:
            plt.show()
            return fig

    def top_global(self, n, no_plot=False):

        import matplotlib.patheffects as PathEffects

        selected_global,important_features = self.top_global_table(n)

        m = len(important_features)

        fig = plt.figure(figsize=(n, n))
        ax = fig.add_axes([0, 0, .8, 1])
        plt.title("Global Correlations",fontsize=15)
        im = ax.imshow(selected_global, vmin=-1, vmax=1, cmap='bwr')
        for i in range(m):
            for j in range(m):
                text = ax.text(j, i, np.around(selected_global[i, j], 3),
                               ha="center", va="center", c='w', fontsize=7)
                text.set_path_effects(
                    [PathEffects.withStroke(linewidth=.5, foreground='black')])

        plt.xticks(np.arange(m), labels=important_features, rotation=45)
        plt.yticks(np.arange(m), labels=important_features, rotation=45)
        cb_ax = fig.add_axes([.85,.1,.1,.8])
        plt.colorbar(im,cax=cb_ax,label="Weighted Pearson Correlation")
        plt.tight_layout()
        if no_plot:
            return fig
        else:
            plt.show()
            return fig

    def html_cross_reference(self, n=10, plot=False, output=None):

        if output is None:
            location = self.html_directory()
        else:
            location = output

        # local_cross,important_local = self.top_local_table(n)
        # global_cross,important_global = self.top_global_table(n)
        #
        # local_html = generate_cross_reference_table(local_cross,important_local)
        # global_html = generate_cross_reference_table(global_cross,important_global)

        print("Generating top local")

        local_cross = self.top_local(n)

        print("Generating top global")
        global_cross = self.top_global(n)

        print(f"Saving cross ref to {location}")

        local_cross.savefig(location + "local_cross.png", bbox_inches='tight', dpi=DPI_SET)
        global_cross.savefig(
            location + "global_cross.png", bbox_inches='tight', dpi=DPI_SET)

        local_html = f'<img class="local_cross" src="{location + "local_cross.png"}" />'
        global_html = f'<img class="global_cross" src="{location + "global_cross.png"}" />'

        return (local_html, global_html)

    def html_cluster_summary(self, n=20, plot=True, output=None):

        location = self.forest.location()
        if output is None:
            html_location = self.html_directory()
        else:
            html_location = output

        # First we read in the template (TO DO improve safety)
        html_string = open(
            location + "/cluster_summary_template_js.html", 'r').read()

        # Reading the file in allows us to both write a summary file somewhere appropriate and return an html string in case we want to do something else

        # This function puts the sister score image in the appropriate location (we discard its return string, not relevant here)

        print("Rendering scores")

        self.html_sister_scores(output=output)
        self.html_sample_scores(output=output)
        # self.html_cross_reference(n=n, output=output)

        with open(html_location + "cluster_summary_template_js.html", 'w') as html_file:
            json_string = js_wrap("attributes", self.json_cluster_summary(n=n))
            html_string = html_string + json_string
            html_file.write(html_string)

        if plot:
            # We ask the OS to open the html file.
            from subprocess import run
            run(["open", html_location + "cluster_summary_template_js.html"])

        # Finally we return the HTML string
        # CAUTION, this contains the whole template file, so it has a bunch of javascript in it.
        return html_string

    def html_feature_means(self, features):
        feature_values = self.feature_means(features)
        html = generate_feature_value_html(features, feature_values)
        return html

    def html_feature_additives(self, features):
        feature_values = self.feature_additives(features)
        html = generate_feature_value_html(features, feature_values)
        return html

    def html_sister_scores(self, output=None):

        if output is None:
            location = self.html_directory()
        else:
            location = output

        forest_coordinates = self.forest.coordinates(no_plot=True)
        sister_scores = self.sister_scores()
        plt.figure()
        plt.title(
            f"Distribution of Samples \nIn {self.name()} (Red) vs Its Sisters (Blue)")
        plt.scatter(forest_coordinates[:, 0], forest_coordinates[:, 1], s=1,
                    alpha=.6, c=sister_scores, norm=DivergingNorm(0), cmap='bwr')
        plt.colorbar(label="Sister Score (Difference in Probability)")
        plt.ylabel("tSNE Coordinates (AU)")
        plt.xlabel("tSNE Coordinates (AU)")
        plt.savefig(location + "sister_map.png",dpi=DPI_SET)

        html = f'<img class="sister_score" src="{location + "sister_map.png"}" />'

        return html

    def html_sample_scores(self, output=None):

        if output is None:
            location = self.html_directory()
        else:
            location = output

        forest_coordinates = self.forest.coordinates(no_plot=True)
        sample_scores = self.sample_scores()
        plt.figure()
        plt.title(f"Frequency of Samples In {self.name()}")
        plt.scatter(
            forest_coordinates[:, 0], forest_coordinates[:, 1], s=1, alpha=.6, c=sample_scores)
        plt.colorbar()
        plt.ylabel("tSNE Coordinates (AU)")
        plt.xlabel("tSNE Coordinates (AU)")
        plt.savefig(location + "score_map.png", dpi=DPI_SET)

        html = f'<img class="score_map" src="{location + "score_map.png"}" />'

        return html

    def sample_cluster_frequency(self, plot=True):
        sample_cluster_labels = self.forest.sample_labels
        sample_counts = self.sample_counts()
        sample_clusters = sorted(list(set(sample_cluster_labels)))
        cluster_counts = []
        for cluster in sample_clusters:
            cluster_mask = sample_cluster_labels == cluster
            cluster_counts.append(np.sum(sample_counts[cluster_mask]))

        if plot:
            plt.figure()
            plt.title("Frequency of sample clusters in leaf cluster")
            plt.bar(np.arange(len(sample_clusters)),
                    cluster_counts, tick_labels=sample_clusters)
            plt.ylabel("Frequency")
            plt.show()

        return sample_clusters, cluster_counts

    def plot_sample_counts(self, **kwargs):
        counts = self.sample_counts()
        plt.figure(figsize=(15, 10))
        plt.scatter(self.forest.coordinates(no_plot=True)[
                    :, 0], self.forest.coordinates(no_plot=True)[:, 1], c=counts, **kwargs)
        plt.colorbar()
        plt.show()

    def plot_sister_scores(self,**kwargs):
        sister_scores = self.sister_scores()
        forest_coordinates = self.forest.coordinates(no_plot=True)
        fig = plt.figure()
        plt.title(
            f"Distribution of Samples \nIn {self.name()} (Red) vs Its Sisters (Blue)")
        plt.scatter(forest_coordinates[:, 0], forest_coordinates[:, 1], s=1,
                    alpha=.6, c=sister_scores, norm=DivergingNorm(0), cmap='bwr')
        plt.colorbar(label="Sister Score (Difference in Probability)")
        plt.ylabel("tSNE Coordinates (AU)")
        plt.xlabel("tSNE Coordinates (AU)")
        return fig

    def probability_enrichment(self):
        enrichment = self.forest.probability_enrichment()
        return enrichment.T[self.id]

    def odds_ratio(self):
        odds_ratios = self.forest.split_cluster_odds_ratios()
        return odds_ratios.T[self.id]
