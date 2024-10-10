import numpy as np
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures

from cdrl.utils.graph_utils import get_int_representations, contains_cycles, edge_list_from_adj_matrix

class ContinuousVarsBICRewardFunction(object):
    """
    Reward function for causal graphs with continuous random variables.
    This is adapted from the RL-BIC codebase. It was slightly refactored and integrated without changing the functionality.
    The original can be consulted under rlbic-src/rewards/Reward_BIC.py.
    """
    name = "bic"

    CYCLIC_GRAPH_REWARD = -100.

    def __init__(self, disc_inst, score_type='BIC', reg_type='LR', penalise_cyclic=True, store_scores=True, **kwargs):
        self.store_scores = store_scores
        if self.store_scores:
            self.cached_scores = {}  # store results
            self.cached_RSS = {}  # store RSS for reuse

        inputdata = disc_inst.inputdata
        self.d = inputdata.shape[1]
        self.inputdata = inputdata
        self.n_samples = inputdata.shape[0]
        self.bic_penalty = np.log(inputdata.shape[0]) / inputdata.shape[0]

        self.penalise_cyclic = penalise_cyclic

        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type

        self.ones = np.ones((inputdata.shape[0], 1), dtype=np.float32)
        self.poly = PolynomialFeatures()

        self.regression_count = 0
        self.bic_evaluation_count = 0

    def calculate_yerr(self, X_train, y_train):
        """Computes the residuals (differences between the predicted and actual values)."""
        self.regression_count += 1
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train)
        else:
            raise ValueError('Regressor not supported')

    # faster than LinearRegression() from sklearn
    def calculate_LR(self, X_train, y_train):
        """Runs a linear regression and computes residuals."""
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        """Runs a quadratic regression and computes residuals."""
        X_train = self.poly.fit_transform(X_train)[:, 1:]
        return self.calculate_LR(X_train, y_train)

    def calculate_GPR(self, X_train, y_train):
        """Runs a Gaussian Process regression and computes residuals."""
        med_w = np.median(pdist(X_train, 'euclidean'))
        gpr = GPR().fit(X_train / med_w, y_train)
        return y_train.reshape(-1, 1) - gpr.predict(X_train / med_w).reshape(-1, 1)

    ####### score calculations
    def calculate_reward_single_graph(self, graph):
        """
        Computes the reward (i.e., score function) for a given DAGState.
        This method iterates through all nodes in the graph and regresses
        each variable on its parents.
        It additionally features a caching mechanism so that regressions for a given (node, parent set) combination
        are stored.
        """
        adj_matrix_ij = graph.get_adjacency_matrix()
        adj_matrix_ji = adj_matrix_ij.T

        if self.penalise_cyclic and contains_cycles(adj_matrix_ji):
            return self.CYCLIC_GRAPH_REWARD

        if self.store_scores:
            graph_to_int, graph_to_int2 = get_int_representations(adj_matrix_ji)
            graph_batch_to_tuple = tuple(graph_to_int2)

            if graph_batch_to_tuple in self.cached_scores:
                score = self.cached_scores[graph_batch_to_tuple]
                return -score

        RSS_ls = []

        for i in range(self.d):
            col = adj_matrix_ji[i]
            if self.store_scores:
                if graph_to_int[i] in self.cached_RSS:
                    RSS_ls.append(self.cached_RSS[graph_to_int[i]])
                    continue

            y_err = self.regress_or_mean(self.inputdata, col, i)
            RSSi = np.sum(np.square(y_err))

            # if the regresors include the true parents, GPR would result in very samll values, e.g., 10^-13
            # so we add 1.0, which does not affect the monotoniticy of the score
            if self.reg_type == 'GPR':
                RSSi += 1.0

            RSS_ls.append(RSSi)

            if self.store_scores:
                self.cached_RSS[graph_to_int[i]] = RSSi

        BIC = self.bic_from_RSS_list(RSS_ls, adj_matrix_ji)

        self.bic_evaluation_count += 1
        score = BIC

        if self.store_scores:
            self.cached_scores[graph_batch_to_tuple] = score

        reward = -score
        return reward

    def regress_or_mean(self, dataset, adj_column, i):
        """
        Performs regression -- if there are no parents, use mean as the prediction.
        """
        # no parents, then simply use mean
        if np.sum(adj_column) < 0.1:
            y_err = dataset[:, i]
            y_err = y_err - np.mean(y_err)

        # run the actual regression and compute the residual.
        else:
            cols_TrueFalse = adj_column > 0.5
            X_train = dataset[:, cols_TrueFalse]
            y_train = dataset[:, i]
            y_err = self.calculate_yerr(X_train, y_train)
        return y_err

    def bic_from_RSS_list(self, RSS_ls, adj_matrix_ji):
        """Computes the BIC for the graph based on a list of residual sums of squares."""
        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls) / self.n_samples + 1e-8) \
                  + np.sum(adj_matrix_ji) * self.bic_penalty / self.d
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls) / self.n_samples + 1e-8)) \
                  + np.sum(adj_matrix_ji) * self.bic_penalty
        return BIC





