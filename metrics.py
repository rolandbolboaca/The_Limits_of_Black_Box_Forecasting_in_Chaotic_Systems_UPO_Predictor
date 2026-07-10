import numpy as np
from scipy.stats import entropy, energy_distance, wasserstein_distance
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from aeon.distances import sbd_distance

class Metrics:
    def __init__(self, eps=1e-10):
        self.eps = eps

        # Metrics list
        self.metrics_array = [
            'mse', 'nmse', 'mae', 'mpe', 'mpe_abs', 'mape', 'smape', 
            'rmspe', 'msle', 'r2', 'cosine_similarity', 'canberra', 'lorentzian',
            'jaccard', 'energy', 'wasserstein', 'kullback_leibler', 
            'entropy', 'js_divergence', 'jeffrey_divergence', 'hellinger', 'sbd'
        ]

        self.metrics_multivariate = [
            'mse', 'nmse', 'mae', 'mpe', 'mpe_abs', 'mape', 'smape', 
            'rmspe', 'msle', 'r2', 'cosine_similarity', 'canberra', 
            'jaccard', 'energy', 'lorentzian', 'sbd'
        ]

        self.metrics_univariate = [
            'kullback_leibler', 'entropy', 'js_divergence', 
            'jeffrey_divergence', 'hellinger', 'wasserstein'
        ]

        self.custom_metrics_multivariate = [
            'mse', 'mae', 'r2', 'energy', 'cosine_similarity', 'canberra', 'lorentzian', 'sbd'
        ]


    def _flatten(self, x):
        return np.ravel(np.array(x, dtype=float))

    def get_pds(self, v1, v2):
        all_data = np.concatenate([v1, v2])
        bins = np.histogram_bin_edges(all_data, bins='fd')

        p, _ = np.histogram(v1, bins=bins, density=True)
        q, _ = np.histogram(v2, bins=bins, density=True)

        p = np.where(p == 0, self.eps, p / np.sum(p))
        q = np.where(q == 0, self.eps, q / np.sum(q))
        return p, q


    def lorentzian_error(self, y_true, y_pred):
        """
        Lorentzian (Cauchy) error metric for time series.
        Equivalent to mean(log(1 + |x_i - y_i|)).
        """

        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

        return np.mean(np.log(1.0 + np.abs(y_true - y_pred)))

    def kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))

    def jeffrey_divergence(self, p, q):
        return self.kl_divergence(p, q) + self.kl_divergence(q, p)

    def js_divergence(self, p, q):
        return jensenshannon(p, q) ** 2

    def hellinger_distance(self, p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    def compute_residuals(self, 
                          observed_data=None, 
                          predicted_data=None,
                          residual_type='mse',
                          average_metrics = []):
        """
        Compute various residuals and similarity metrics between observed and predicted data.
        Automatically handles univariate and multivariate data.

        Relevant paper: https://arxiv.org/html/2412.20574v1
        Compute residuals and metrics

            - The lock-step measures return the individual errors and the mean (or other aggregation) of the errors (the metric value)
            - The probability and disimilarity distance measures return the aggregation of the errors (the metric value)
            - Will be extended in the future.

        """

        obs = np.array(observed_data, dtype=float)
        pred = np.array(predicted_data, dtype=float)

        if obs.shape != pred.shape:
            raise ValueError("Observed and predicted data must have the same shape.")

        residual_type = residual_type.lower()

        # if obs.ndim > 1 and residual_type in [
        #     'kullback_leibler', 'entropy', 'js_divergence', 
        #     'jeffrey_divergence', 'hellinger', 'wasserstein']:
        #         raise ValueError(f"{residual_type} is only supported for univariate data, or at least it makes no sense.")
        
        obs_flat = self._flatten(obs)
        pred_flat = self._flatten(pred)

        #  Lock-step metrics 
        if residual_type == 'mse':
            errors = (obs - pred) ** 2
            return errors, np.mean(errors)

        elif residual_type == 'nmse':
            squared_errors = (obs - pred) ** 2
            obs_squared = obs ** 2
            return squared_errors, np.mean(squared_errors / (obs_squared + self.eps))

        elif residual_type == 'mae':
            abs_errors = np.abs(obs - pred)
            return abs_errors, np.mean(abs_errors)

        elif residual_type == 'mpe':
            pe_errors = (obs - pred) / (obs + self.eps)
            return pe_errors, np.mean(pe_errors)

        elif residual_type == 'mpe_abs':
            mpe_abs_errors = np.abs((obs - pred) / (obs + self.eps))
            return mpe_abs_errors, np.mean(mpe_abs_errors)

        elif residual_type == 'mape':
            mape_errors = np.abs((obs - pred) / (obs + self.eps))
            return mape_errors, np.mean(mape_errors)

        elif residual_type == 'smape':
            smape_errors = 2 * np.abs(obs - pred) / (np.abs(obs) + np.abs(pred) + self.eps)
            return smape_errors, np.mean(smape_errors)

        elif residual_type == 'rmspe':
            rmspe_errors = ((obs - pred) / (obs + self.eps)) ** 2
            return rmspe_errors, np.sqrt(np.mean(rmspe_errors))

        elif residual_type == 'msle':
            msle_errors = (np.log1p(pred) - np.log1p(obs)) ** 2
            return msle_errors, np.mean(msle_errors)

        elif residual_type == 'r2':
            return None, r2_score(obs_flat, pred_flat)

        elif residual_type == 'cosine_similarity':
            sim = cosine_similarity(obs.reshape(1, -1), pred.reshape(1, -1))[0, 0]
            return None, sim

        elif residual_type == 'canberra':
            return None, distance.canberra(obs_flat, pred_flat)

        elif residual_type == 'jaccard':
            return None, distance.jaccard(obs_flat > 0, pred_flat > 0)
        
        elif residual_type == 'lorentzian':
            return None, self.lorentzian_error(obs_flat, pred_flat)

        # Shape-based distance
        # https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.distances.sbd_distance.html
        elif residual_type == 'sbd':
            return None, sbd_distance(obs_flat, pred_flat)
        
        #  Probability-based metrics 
        elif residual_type == 'entropy':
            p, q = self.get_pds(obs_flat, pred_flat)
            return None, entropy(p, q)

        elif residual_type == 'js_divergence':
            p, q = self.get_pds(obs_flat, pred_flat)
            return None, self.js_divergence(p, q)

        elif residual_type == 'kullback_leibler':
            p, q = self.get_pds(obs_flat, pred_flat)
            return None, self.kl_divergence(p, q)

        elif residual_type == 'jeffrey_divergence':
            p, q = self.get_pds(obs_flat, pred_flat)
            return None, self.jeffrey_divergence(p, q)

        elif residual_type == 'hellinger':
            p, q = self.get_pds(obs_flat, pred_flat)
            return None, self.hellinger_distance(p, q)

        elif residual_type == 'energy':
            return None, energy_distance(obs_flat, pred_flat)

        elif residual_type == 'wasserstein':
            return None, wasserstein_distance(obs_flat, pred_flat)

        elif residual_type == "average":
            m1 = (obs - pred) ** 2
            m2 = self.sbd_distance(obs_flat, pred_flat)
            m3 = self.energy_distance(obs_flat, pred_flat)
            return (m1 + m2 + m3) / 3
        
        else:
            raise ValueError(f"Unknown residual_type: {residual_type}. Possible values: {self.metrics_array}")

    def compute_all_residuals(self, observed_data, predicted_data):
        results = {}
        for residual_type in self.metrics_array:
            try:
                _, val = self.compute_residuals(observed_data, predicted_data, residual_type)
                results[residual_type] = val
            except ValueError as e:
                results[residual_type] = str(e)
        return results

    def compute_multivariate_residuals(self, observed_data, predicted_data):
        results = {}
        for residual_type in self.metrics_multivariate:
            try:
                _, val = self.compute_residuals(observed_data, predicted_data, residual_type)
                results[residual_type] = val
            except ValueError as e:
                results[residual_type] = str(e)
        return results
    
    def compute_univariate_residuals(self, observed_data, predicted_data):

        results_per_variable = {}

        i = 0
        for obs, pred in zip(observed_data, predicted_data):

            results = {}
            for residual_type in self.metrics_univariate:
                try:
                    _, val = self.compute_residuals(obs, pred, residual_type)
                    results[residual_type] = val
                except ValueError as e:
                    results[residual_type] = str(e)

            i += 1
            results_per_variable[i] = results

        return results
    
    def compute_custom_multivariate(self, observed_data, predicted_data):
        results = {}
        for residual_type in self.custom_metrics_multivariate:
            try:
                _, val = self.compute_residuals(observed_data, predicted_data, residual_type)
                results[residual_type] = val
            except ValueError as e:
                results[residual_type] = str(e)
        return results