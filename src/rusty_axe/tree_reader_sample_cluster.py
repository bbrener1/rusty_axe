import numpy as np


class SampleCluster:

    def __init__(self, forest, samples, id):
        self.id = id
        self.samples = samples
        self.forest = forest

    def name(self):
        if hasattr(self, 'stored_name'):
            return self.stored_name
        else:
            return str(self.id)

    def set_name(self, name):
        self.stored_name = name

    def mask(self):
        mask = np.zeros(len(self.forest.samples), dtype=bool)
        mask[self.samples] = True
        return mask

    def median_feature_values(self):
        return np.median(self.forest.output[self.samples], axis=0)

    def mean_feature_values(self):
        return np.mean(self.forest.output[self.samples], axis=0)

    def increased_features(self, n=50, plot=True):
        initial_means = np.mean(self.forest.output)
        current_means = self.mean_feature_values()

        difference = current_means - initial_means
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.forest.output_features)[feature_order]
        ordered_difference = difference[feature_order]

        if plot:
            plt.figure(figsize=(10, 8))
            plt.title("Upregulated Genes")
            plt.scatter(np.arange(n), ordered_difference[-n:])
            plt.xlim(0, n)
            plt.xlabel("Gene Symbol")
            plt.ylabel("Increase (LogTPM)")
            plt.xticks(np.arange(
                n), ordered_features[-n:], rotation=45, verticalalignment='top', horizontalalignment='right')
            plt.show()

        return ordered_features, ordered_difference

    def decreased_features(self, n=50, plot=True):
        initial_means = np.mean(self.forest.output)
        current_means = self.mean_feature_values()

        difference = current_means - initial_means
        feature_order = np.argsort(difference)
        ordered_features = np.array(self.forest.output_features)[feature_order]
        ordered_difference = difference[feature_order]

        if plot:
            plt.figure(figsize=(10, 8))
            plt.title("Upregulated Genes")
            plt.scatter(np.arange(n), ordered_difference[:n])
            plt.xlim(0, n)
            plt.xlabel("Gene Symbol")
            plt.ylabel("Increase (LogTPM)")
            plt.xticks(np.arange(
                n), ordered_features[:n], rotation=45, verticalalignment='top', horizontalalignment='right')
            plt.show()

        return ordered_features, ordered_difference

    def logistic_features(self, n=50):

        from sklearn.linear_model import LogisticRegression

        mask = self.mask()

        scaled = sklearn.preprocessing.scale(self.forest.input)

        model = LogisticRegression().fit(scaled, mask)

        coefficient_sort = np.argsort(model.coef_[0])

        sorted_features = self.forest.input_features[coefficient_sort][-n:]
        sorted_coefficients = model.coef_[0][coefficient_sort][-n:]

        return sorted_features, sorted_coefficients

    def leaf_encoding(self):
        leaves = self.forest.leaves()
        encoding = self.forest.node_sample_encoding(leaves)
        encoding = encoding[self.samples]
        return encoding

    def leaf_counts(self):
        encoding = self.leaf_encoding()
        return np.sum(encoding, axis=0)

    def leaf_cluster_frequency(self, plot=True):
        leaf_counts = self.leaf_counts()
        leaf_cluster_labels = self.forest.leaf_labels
        leaf_clusters = sorted(list(set(leaf_cluster_labels)))
        leaf_cluster_counts = []
        for leaf_cluster in leaf_clusters:
            cluster_mask = leaf_cluster_labels == leaf_cluster
            leaf_cluster_counts.append(np.sum(leaf_counts[cluster_mask]))
        if plot:
            plt.figure()
            plt.title(
                f"Distribution of Leaf Clusters in Sample Cluster {self.name()}")
            plt.bar(np.arange(len(leaf_clusters)), leaf_cluster_counts,)
            plt.ylabel("Frequency")
            plt.xlabel("Leaf Cluster")
            plt.xticks(np.arange(len(leaf_clusters)), leaf_clusters)
            plt.show()

        return leaf_clusters, leaf_cluster_counts

    def feature_median(self, feature):
        fi = self.forest.truth_dictionary.feature_dictionary[feature]
        vector = self.forest.output[self.samples][:, fi]
        return np.median(vector)

    def feature_mean(self, feature):
        fi = self.forest.truth_dictionary.feature_dictionary[feature]
        vector = self.forest.output[self.samples][:, fi]
        return np.mean(vector)
