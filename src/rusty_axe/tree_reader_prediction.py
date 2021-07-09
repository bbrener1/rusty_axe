import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import entropy
from scipy.stats import ks_2samp
from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from scipy.stats import t,iqr
from scipy.spatial.distance import cdist,pdist

# from tree_reader_utils import jackknife_variance

mpl.rcParams['figure.dpi'] = 100


class Prediction:

    def __init__(self, forest, matrix):
        self.forest = forest
        self.matrix = matrix
        self.mode = None
        self.nse = None
        self.nme = None
        self.nae = None
        self.smc = None
        self.nsr2 = None
        self.nfr2 = None
        self.factors = None

    def node_sample_encoding(self):
        if self.nse is None:
            self.nse = self.forest.predict_node_sample_encoding(
                self.matrix, leaves=False)
        return self.nse

    def node_mean_encoding(self):
        if self.nme is None:
            self.nme = self.forest.mean_matrix(self.forest.nodes())
        return self.nme

    def node_additive_encoding(self):
        if self.nae is None:
            self.nae = self.forest.mean_additive_matrix(self.forest.nodes())
        return self.nae

    def node_sample_r2(self):
        truth = self.matrix
        self.node_mean_encoding()
        self.node_sample_encoding()
        if self.nsr2 is None:
            print("Computing Node-Sample R2")
            self.nsr2 = np.zeros(self.nse.shape)
            for node in self.forest.nodes():
                if node.index % 100 == 0:
                    print(f"Node {node.index}",end='\r')
                node_prediction = self.nme[node.index]
                selection = truth[self.nse[node.index]]
                residuals = selection - node_prediction
                self.nsr2[node.index][self.nse[node.index]] = np.sum(np.power(residuals,2),axis=1)
            print("")
        return self.nsr2

    def node_feature_r2(self):
        truth = self.matrix
        self.node_mean_encoding()
        self.node_sample_encoding()
        if self.nfr2 is None:
            print("Computing Node-Sample R2")
            self.nfr2 = np.zeros(self.nme.shape)
            for node in self.forest.nodes():
                if node.index % 100 == 0:
                    print(f"Node {node.index}",end='\r')
                node_prediction = self.nme[node.index]
                selection = truth[self.nse[node.index]]
                residuals = selection - node_prediction
                self.nfr2[node.index] = np.sum(np.power(residuals,2),axis=0)

        return self.nfr2


    def additive_prediction(self, depth=8):
        encoding = self.node_sample_encoding().T
        feature_predictions = self.node_additive_encoding().T
        prediction = np.dot(encoding.astype(dtype=int), feature_predictions)
        prediction /= len(self.forest.trees)

        return prediction

    def mean_prediction(self,mask = None):
        if mask is None:
            mask = self.forest.leaf_mask()
        encoding_prediction = self.node_sample_encoding()[mask].T
        feature_predictions = self.node_mean_encoding()[mask]
        scaling = np.dot(encoding_prediction,
                         np.ones(feature_predictions.shape))

        prediction = np.dot(encoding_prediction, feature_predictions) / scaling
        prediction[scaling == 0] = 0
        return prediction

    def observed_means(self,nodes = None):
        if nodes is None:
            nodes = self.forest.nodes()
        nse = self.node_sample_encoding()
        node_mask = np.zeros(nse.shape[0],dtype=bool)
        node_mask[[n.index for n in nodes]] = True
        nse = nse[node_mask]
        means = np.array([np.mean(self.matrix[mask],axis=0) for mask in nse])
        populations = np.sum(nse,axis=1)
        means[populations == 0] = 0
        return means


    def observed_marginal(self,nodes = None):
        if nodes is None:
            nodes = self.forest.nodes()
        nse = self.node_sample_encoding()

        marginal = np.zeros((len(nodes),len(self.forest.output_features)))

        for i,node in enumerate(nodes):
            if node.parent is not None:
                mask = nse[node.index]
                parent_mask = nse[node.parent.index]
                if np.sum(mask.astype(dtype=int)) > 0 and np.sum(parent_mask.astype(dtype=int)) > 0:
                    nm = np.mean(self.matrix[mask],axis=0)
                    pnm = np.mean(self.matrix[parent_mask],axis=0)
                    node_marginal = nm - pnm
                    marginal[i] = node_marginal
                else: marginal[i] = 0
        return marginal

    def prediction(self, mode=None):

        if mode is None:
            mode = self.mode
        if mode is None:
            mode = "additive_mean"
        self.mode = mode

        if mode == "additive_mean":
            prediction = self.additive_prediction()
        elif mode == "mean":
            prediction = self.mean_prediction()
        else:
            raise Exception(f"Not a valid mode {mode}")

        return prediction

    def residuals(self, truth=None, mode=None):

        if truth is None:
            truth = self.matrix

        prediction = self.prediction(mode=mode)
        residuals = truth - prediction

        return residuals

    def null_residuals(self, truth=None):

        if truth is None:
            truth = self.matrix

        centered_truth = truth - np.mean(truth, axis=0)

        return centered_truth

    def null_r2(self):

        null_residuals = self.null_residuals()

        return np.sum(np.power(null_residuals,2),axis=0)

    def node_residuals(self, node, truth=None):

        if truth is None:
            truth = self.matrix

        sample_predictions = self.node_sample_encoding()[node.index]
        feature_predictions = self.node_mean_encoding()[node.index]
        residuals = truth[sample_predictions] - feature_predictions

        return residuals

    def node_feature_r2(self,node,truth=None):

        if truth is None:
            truth = self.matrix

        sample_predictions = self.node_sample_encoding()[node.index]
        feature_predictions = self.node_mean_encoding()[node.index]
        true_sum = np.sum(np.power(truth[sample_predictions],2),axis=0)
        r2 = true_sum - (sample_predictions * np.sum(sample_predictions.astype(dtype=int)))


        return r2


    def node_fraction(self, node):

        self_samples = np.sum(self.node_sample_encoding()[node.index])
        if node.parent is None:
            parent_samples = self_samples
        else:
            parent_samples = np.sum(self.node_sample_encoding()[node.parent.index])
        if parent_samples > 0:
            return float(self_samples)/float(parent_samples)
        else:
            return 0

    def node_mse(self,node):

        residuals = self.node_residuals(node)

        return np.sum(np.power(residuals,2)) / (residuals.shape[0] * residuals.shape[1])

    def node_residual_doublet(self,node):

        truth = self.matrix

        sample_predictions = self.node_sample_encoding()[node.index]

        self_predictions = self.node_mean_encoding()[node.index]
        self_residuals = truth[sample_predictions] - self_predictions

        if node.parent is not None:
            parent_predictions = self.node_mean_encoding()[node.parent.index]
        else:
            parent_predictions = np.zeros(self_predictions.shape)
        parent_residuals = truth[sample_predictions] - parent_predictions

        return self_residuals,parent_residuals

    def node_r2_doublet(self,node):

        sample_mask = self.node_sample_encoding()[node.index]

        self_r2 = np.sum(self.node_sample_r2()[node.index][sample_mask])

        if node.parent is not None:
            parent_r2 = np.sum(self.node_sample_r2()[node.parent.index][sample_mask])
        else:
            parent_r2 = 0

        return self_r2,parent_r2

    def node_feature_error(self,node):
        residuals = self.node_residuals(node)
        return np.sum(np.power(residuals,2),axis=0)

    def factor_feature_error(self,factor):

        self_total_error = np.zeros(len(self.forest.output_features))
        parent_total_error = np.zeros(len(self.forest.output_features))

        for i,node in enumerate(factor.nodes):
            if i % 10 == 0:
                print(f"{i}/{len(factor.nodes)}",end='\r')

            self_residuals,parent_residuals = self.node_residual_doublet(node)

            self_total_error += np.sum(np.power(self_residuals,2),axis=0)
            parent_total_error += np.sum(np.power(parent_residuals,2),axis=0)

        print("\n",end='')

        return self_total_error,parent_total_error


    def factor_total_error(self,factor):

        self_total_error = 0
        parent_total_error = 0

        for i,node in enumerate(factor.nodes):
            if i % 10 == 0:
                print(f"{i}/{len(factor.nodes)}",end='\r')

            self_r2,parent_r2 = self.node_r2_doublet(node)
            self_total_error += self_r2
            parent_total_error += parent_r2

        print("\n",end='')

        return self_total_error,parent_total_error

    def factor_mse(self,factor):

        node_mses = np.array([self.node_mse(n) for n in factor.nodes])

        # Filter for empty nodes
        node_mses = node_mses[np.isfinite(node_mses)]

        mse = np.mean(node_mses)
        variance = np.var(node_mses)
        f_iqr = iqr(node_mses,rng=(5,95))
        #
        # print(f"Jackknife debug:{n},{mse_estimate},{variance_estimate}")

        return mse,variance,f_iqr


    def jackknife_factor_mse(self,factor):

        node_mses = np.array([self.node_mse(n) for n in factor.nodes])

        # Filter for empty nodes
        node_mses = node_mses[np.isfinite(node_mses)]

        n = len(node_mses)

        total = np.sum(node_mses)
        mse_estimate = total / n

        excluded_sum = total - node_mses
        excluded_means = excluded_sum / (n - 1)
        variance_estimate = ((n - 1) / n) * np.sum(np.power(excluded_means - mse_estimate,2))
        #
        # print(f"Jackknife debug:{n},{mse_estimate},{variance_estimate}")

        return mse_estimate,variance_estimate


    def compare_factor_fractions(self,other,factor,plot=False):

        print(f"Comparing Split Fraction for Factor {factor.name()}")

        self_fractions = np.array([self.node_fraction(n) for n in factor.nodes])
        other_fractions = np.array([other.node_fraction(n) for n in factor.nodes])

        if plot:
            plt.figure()
            plt.hist(self_fractions,density=True,bins=20,label='Self',alpha=.5)
            plt.hist(other_fractions,density=True,bins=20,label='Other',alpha=.5)
            plt.legend()
            plt.xlabel("Fraction")
            plt.ylabel("Frequency")
            plt.show()

        self_mean = np.mean(self_fractions)
        other_mean = np.mean(other_fractions)

        print(f"Self: {self_mean}")
        print(f"Other: {other_mean}")

        result = mannwhitneyu(self_fractions,other_fractions)
        print(result)

        return self_mean,other_mean,result
    #
    # def compare_factor_features(self,other,factor):


    def compare_factor_residuals(self,other,factor):

        print(f"Comparing residuals for Factor {factor.name()}")

        self_factor_mse,self_factor_mse_variance,self_factor_iqr = self.factor_mse(factor)
        other_factor_mse,other_factor_mse_variance,other_factor_iqr = other.factor_mse(factor)

        # self_factor_mse,self_factor_mse_variance = self.jackknife_factor_mse(factor)
        # other_factor_mse,other_factor_mse_variance = other.jackknife_factor_mse(factor)

        self_mse_std = np.sqrt(self_factor_mse_variance)
        factor_z = (self_factor_mse - other_factor_mse) / self_mse_std
        factor_p = t.pdf(factor_z,len(factor.nodes) - 1)

        print(f"Self Factor MSE:{self_factor_mse}, +/- {self_factor_mse}")
        print(f"Self Factor MSE 90% interval: {self_factor_iqr}")
        print(f"Other Factor MSE:{other_factor_mse}")

        return (factor_z,factor_p)

    def compare_factor_fvu(self,other,factor,plot=False):

        print(f"Estimating FVU for Factor {factor.name()}")

        self_doublets = [self.node_r2_doublet(n) for n in factor.nodes]
        other_doublets = [other.node_r2_doublet(n) for n in factor.nodes]

        self_node_cod = np.array([1-(n/p) for (n,p) in self_doublets])
        other_node_cod = np.array([1-(n/p) for (n,p) in other_doublets])
        self_node_cod = self_node_cod[np.isfinite(self_node_cod)]
        other_node_cod = other_node_cod[np.isfinite(other_node_cod)]

        cod_range = np.quantile(self_node_cod,.05),np.quantile(other_node_cod,.95)

        mwu = mannwhitneyu(self_node_cod, other_node_cod)
        print(mwu)

        if plot:
            plt.figure()
            plt.title("Distritbution of Node CODs")
            plt.hist(self_node_cod ,density=True,bins=20,label='Self',alpha=.5)
            plt.hist(other_node_cod,density=True,bins=20,label='Other',alpha=.5)
            plt.legend()
            plt.xlabel("Fraction")
            plt.ylabel("Frequency")
            plt.show()

        self_self,self_parent = self.factor_total_error(factor)
        other_self,other_parent = other.factor_total_error(factor)

        self_fvu = np.sum(self_self)/ np.sum(self_parent)
        other_fvu = np.sum(other_self)/ np.sum(other_parent)

        print(f"Self FVU: {self_fvu}")
        print(f"Other FVU: {other_fvu}")

        print(f"Self COD: {1-self_fvu} {cod_range}")
        print(f"Other COD: {1-other_fvu}")

        return (self_fvu, other_fvu,mwu)

    def compare_factor_values(
            self,
            other,
            factor,
            mode="mann_whitney_u",
            no_plot=False,
            bins=100,
            log=True
        ):

        bin_interval = 2.0 / bins

        print(f"Now comparing values for Factor {factor.name()}:")

        own_f = self.factor_matrix()[:,factor.id]
        other_f = other.factor_matrix()[:,factor.id]

        own_hist = np.histogram(
            own_f, bins=np.arange(-1, 1, bin_interval))[0] + 1
        other_hist = np.histogram(
            other_f, bins=np.arange(-1, 1, bin_interval))[0] + 1
        own_prob = own_hist / np.sum(own_hist)
        other_prob = other_hist / np.sum(other_hist)
        forward_entropy = entropy(own_prob, qk=other_prob)
        reverse_entropy = entropy(other_prob, qk=own_prob)
        symmetric_entropy = (forward_entropy + reverse_entropy) / 2
        print(f"Entropy: {symmetric_entropy}")

        if not no_plot:
            own_log_prob = np.log(own_hist / np.sum(own_hist))
            other_log_prob = np.log(other_hist / np.sum(other_hist))

            lin_min = np.min(
                [np.min(own_log_prob), np.min(other_log_prob)])

            plt.figure(figsize=(5, 5))
            plt.title(f"Factor {factor.name()} Comparison")
            plt.scatter(own_log_prob, other_log_prob,
                        c=np.arange(-1, 1, bin_interval)[:-1], cmap='seismic')
            plt.plot([0, lin_min], [0, lin_min], color='red', alpha=.5)
            plt.xlabel("Factor Frequency, Self (Log Probability)")
            plt.ylabel("Factor Frequency, Other (Log Probability)")
            plt.colorbar(label="Factor Value")
            plt.show()


        if mode == 'mann_whitney_u':
            mwu = mannwhitneyu(own_f, other_f)
            print(f"Mann-Whitney U: {mwu}")
            return mwu,symmetric_entropy
        elif mode == 'kolmogorov_smirnov':
            ks = ks_2samp(own_f, other_f)
            print(f"Kolmogorov-Smirnov: {ks}")
            return ks,symmetric_entropy
        else:
            raise Exception(f"Mode not recognized: {mode}")

    def compare_factor_marginals(self,other,factor,metric='cosine'):

        self_marginal = self.observed_marginal(nodes=factor.nodes)
        other_marginal = other.observed_marginal(nodes=factor.nodes)

        self_mean_marginal = np.mean(self_marginal,axis=0)
        other_mean_marginal = np.mean(other_marginal,axis=0)

        own_distances = cdist(self_mean_marginal.reshape([1,-1]),self_marginal,metric=metric)[0]
        other_distances = cdist(self_mean_marginal.reshape([1,-1]),other_marginal,metric=metric)[0]

        ab_max = np.max([np.max(np.abs(self_mean_marginal)),np.max(np.abs(other_mean_marginal))])

        plt.figure()
        plt.title("Marginal Feature Gain, Self vs Other")
        plt.scatter(self_mean_marginal,other_mean_marginal)
        plt.plot([-ab_max,ab_max],[-ab_max,ab_max],color='red',label="Slope 1 (Identical)")
        plt.legend()
        plt.xlabel("Self")
        plt.ylabel("Other")
        plt.show()

        plt.figure()
        plt.title("Distances To Mean Factor Marginal")
        plt.hist(own_distances,bins=50,alpha=.5,density=True,label='Self')
        plt.hist(other_distances,bins=50,alpha=.5,density=True,label='Other')
        plt.legend()
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.show()

        return self_mean_marginal,other_mean_marginal

    def sample_clusters(self):

        if self.smc is None:

            leaf_mask = self.forest.leaf_mask()
            encoding_prediction = self.node_sample_encoding()[leaf_mask].T
            leaf_means = np.array([l.sample_cluster_means()
                                   for l in self.forest.leaves()])
            scaling = np.dot(encoding_prediction,
                             np.ones(leaf_means.shape))

            prediction = np.dot(encoding_prediction, leaf_means) / scaling
            prediction[scaling == 0] = 0

            self.smc = np.argmax(prediction, axis=1)

        return self.smc

    def factor_matrix(self):
        if self.factors is None:
            predicted_encoding = self.node_sample_encoding()
            predicted_factors = np.zeros(
                (self.matrix.shape[0], len(self.forest.split_clusters)))
            predicted_factors[:, 0] = 1.
            for i in range(1, len(self.forest.split_clusters[0:])):
                predicted_factors[:, i] = self.forest.split_clusters[i].predict_sister_scores(
                    predicted_encoding)
            self.factors = predicted_factors
        return self.factors

    def compare_sample_clusters(self, other):

        self_samples = self.sample_clusters()
        other_samples = other.sample_clusters()

        plt.figure()
        plt.title("Sample Cluster Frequency, Self vs Other")
        plt.xlabel("Cluster")
        plt.ylabel("Frequency")
        plt.xticks(np.arange(len(self.forest.sample_clusters)))
        plt.hist(self_samples, alpha=.5, density=True, label="Self",
                 bins=np.arange(len(self.forest.sample_clusters) + 1))
        plt.hist(other_samples, alpha=.5, density=True, label="Other",
                 bins=np.arange(len(self.forest.sample_clusters) + 1))
        plt.legend()
        plt.show()
        pass

    def compare_factors(self, other, bins=100):

        fvu_deltas = []
        factor_ps = []

        factor_mwus = []
        factor_symmetric_entropies = []

        for i,factor_object in enumerate(self.forest.split_clusters):

            if i == 0:
                continue

            print("#########################################")
            print(f"Factor {factor_object.name()}")
            print("#########################################")

            self.compare_factor_means(other,factor_object)
            #
            # factor_z,factor_p = self.compare_factor_residuals(other,factor_object)
            #
            # print(f"Student's T: Test Statistic = {factor_z}, p = {factor_p}")
            #
            # factor_ps.append(factor_p)
            #
            # self_fvu,other_fvu = self.compare_factor_fvu(other,factor_object)
            # fvu_deltas.append(other_fvu - self_fvu)
            #
            # mwu,symmetric_entropy = self.compare_factor_values(other,factor_object,bins=bins)
            #
            # factor_mwus.append(mwu)
            # factor_symmetric_entropies.append(symmetric_entropy)
            #
            # fraction_mwu = self.compare_factor_fractions(other,factor_object)

        # result = {
        #     "P values":factor_ps,
        #     "FVU Deltas":fvu_deltas,
        #     "Mann-Whitney U":factor_mwus,
        #     "Symmetric Entropy":factor_symmetric_entropies,
        # }

        # return result

    def compare_factor_means(
                self,
                other,
                factor,
                plot=['scatter'],
                metric = 'euclidean'
            ):


            print(f"Now comparing factor means {factor.name()}:")

            own_means = self.observed_means(nodes=factor.nodes)
            other_means = other.observed_means(nodes=factor.nodes)

            own_meta_means = np.mean(own_means,axis=0)
            other_meta_means = np.mean(other_means,axis=0)

            own_distances = cdist(own_meta_means.reshape([1,-1]),own_means,metric=metric)[0]
            other_distances = cdist(own_meta_means.reshape([1,-1]),other_means,metric=metric)[0]

            own_mean_distance = np.mean(own_distances)
            other_mean_distance = np.mean(other_distances)

            rank = np.sum((own_distances < other_mean_distance).astype(dtype=int))/len(own_distances)

            if 'debug' in plot:
                print(own_distances)
                print(other_distances)

            if 'means' in plot:
                print("Own means")
                print(own_meta_means)
                print("Other means")
                print(other_meta_means)

            if 'tests' in plot:
                print(ks_2samp(own_distances,other_distances))
                print(mannwhitneyu(own_distances,other_distances))

            if 'rank' in plot:

                print(f"Mean Distance: {own_mean_distance}")
                print(f"Mean of Others To Center: {other_mean_distance}")
                print(f"Rank: {rank}")
                print(f"Distance between means: {cdist(own_meta_means.reshape([1,-1]),other_meta_means.reshape([1,-1]),metric=metric)[0,0]}")


            if 'scatter' in plot:

                plt.figure(figsize=(5, 5))
                plt.title(f"Factor {factor.name()} Mean Comparison")
                plt.scatter(own_meta_means,other_meta_means)
                plt.xlabel("Own Means")
                plt.ylabel("Other Means")
                plt.plot([-.5,.5],[-.5,.5],color='red')
                plt.show()

            if 'distance' in plot:

                plt.figure(figsize=(5, 5))
                plt.title(f"Factor {factor.name()} Distances")
                plt.hist(own_distances.flatten(),bins=100,density=True,label="Own Distances", alpha=.5)
                plt.hist(other_distances.flatten(),bins=100,density=True,label="Other Distances", alpha=.5)
                plt.legend()
                plt.show()

    def prediction_report(self, truth=None, n=10, mode="additive_mean", no_plot=False):

        null_square_residuals = np.power(self.null_residuals(truth=truth), 2)
        null_residual_sum = np.sum(null_square_residuals)

        forest_square_residuals = np.power(self.residuals(truth=truth), 2)
        predicted_residual_sum = np.sum(forest_square_residuals)

        unexplained = predicted_residual_sum / null_residual_sum

        print(f"Fraction Unexplained:{unexplained}")

        # Add one here to avoid divisions by zero, but this is bad
        # Need better solution

        null_feature_residuals = np.sum(null_square_residuals, axis=0) + 1
        forest_feature_residuals = np.sum(forest_square_residuals, axis=0) + 1

        features_explained = forest_feature_residuals / null_feature_residuals

        if not no_plot:
            plt.figure()
            plt.title("Distribution of Target Coefficients of Determination")
            plt.hist(features_explained, bins=np.arange(0, 1, .05), log=True)
            plt.xlabel("CoD")
            plt.ylabel("Frequency")
            plt.show()

        feature_sort = np.argsort(features_explained)

        print(
            (self.forest.output_features[feature_sort[:n]], features_explained[feature_sort[:n]]))
        print((self.forest.output_features[feature_sort[-n:]],
               features_explained[feature_sort[-n:]]))

        null_sample_residuals = np.sum(null_square_residuals, axis=1) + 1
        forest_sample_residuals = np.sum(forest_square_residuals, axis=1) + 1

        samples_explained = forest_sample_residuals / null_sample_residuals

        sample_sort = np.argsort(samples_explained)

        print(sample_sort[:n], samples_explained[sample_sort[:n]])
        print(sample_sort[-n:], samples_explained[sample_sort[-n:]])

        if not no_plot:
            plt.figure()
            plt.title("Distribution of Sample Coefficients of Determination")
            plt.hist(samples_explained, bins=np.arange(0, 1, .05), log=True)
            plt.xlabel("CoD")
            plt.ylabel("Frequency")
            plt.show()

        return features_explained, samples_explained

    def feature_mse(self, truth=None, mode='additive_mean'):

        residuals = self.residuals(truth=truth, mode=mode)
        mse = np.mean(np.power(residuals, 2), axis=0)

        return mse

    def jackknife_feature_mse_variance(self, mode='additive_mean'):

        squared_residuals = np.power(self.residuals(mode=mode), 2)
        residual_sum = np.sum(squared_residuals, axis=0)
        excluded_sum = residual_sum - squared_residuals
        excluded_mse = excluded_sum / (squared_residuals.shape[0] - 1)
        jackknife_variance = np.var(
            excluded_mse, axis=0) * (squared_residuals.shape[0] - 1)

        return jackknife_variance


    def feature_remaining_error(self, truth=None, mode='additive_mean'):

        null_square_residuals = np.power(self.null_residuals(truth=truth), 2)
        null_residual_sum = np.sum(null_square_residuals, axis=0)

        forest_square_residuals = np.power(self.residuals(truth=truth), 2)
        predicted_residual_sum = np.sum(forest_square_residuals, axis=0)

        remaining = predicted_residual_sum / null_residual_sum

        return remaining

    def feature_coefficient_of_determination(self, truth=None, mode='additive_mean'):
        remaining_error = self.feature_remaining_error(truth=truth, mode=mode)
        return 1 - remaining_error

    def compare_feature_residuals(self, other, mode='rank_sum', no_plot=True):


        self_residuals = self.residuals()
        other_residuals = other.residuals()

        if mode == 'rank_sum':
            results = [ranksums(self_residuals[:, i], other_residuals[:, i])
                       for i in range(self_residuals.shape[1])]
        elif mode == 'mann_whitney_u':
            results = [mannwhitneyu(self_residuals[:, i], other_residuals[:, i])
                       for i in range(self_residuals.shape[1])]
        elif mode == 'kolmogorov_smirnov':
            results = [ks_2samp(self_residuals[:, i], other_residuals[:, i])
                       for i in range(self_residuals.shape[1])]

        elif mode == 'mse_delta':

            self_mse = self.feature_mse()
            other_mse = other.feature_mse()

            delta_mse = self_mse - other_mse

            jackknife_std = np.sqrt(self.jackknife_feature_mse_variance())
            jackknife_z = delta_mse / jackknife_std

            prob = t.pdf(jackknife_z, len(self.forest.samples) - 1)

            results = list(zip(jackknife_z, prob))

        elif mode == 'cod_delta':

            print("WARNING")
            print("NO SIGNFIFICANCE SCORE IS PROVIDED FOR DIFFERENCE IN COD")

            self_cod = self.feature_coefficient_of_determination()
            other_cod = other.feature_coefficient_of_determination()

            delta_cod = self_cod - other_cod

            results = list(zip(delta_cod, np.ones(len(delta_cod))))

        else:
            raise Exception(f"Did not recognize mode:{mode}")

        if not no_plot:
            plt.figure()
            plt.title("Distribution of Test Statistics")
            plt.hist([test for test, p in results], log=True, bins=50)
            plt.xlabel("Test Statistic")
            plt.ylabel("Frequency")
            plt.show()

            plt.figure()
            plt.title("Distribution of P Values")
            plt.hist([p for test, p in results], log=True, bins=50)
            plt.xlabel("P Value")
            plt.ylabel("Frequency")
            plt.show()

        return results
