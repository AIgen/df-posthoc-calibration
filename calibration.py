import numpy as np
from utils import *

class HB_binary(object):
    def __init__(self):
        ### Hyperparameters
        self.delta = 1e-10
        self.n_bins = None

        ### Parameters of histogram binning that are to be learnt 
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None

        ### Some internal functions
        self.fitted = False
        
    def fit(self, y_score, y):
        assert(self.n_bins is not None), "Number of bins has to be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert(y_score.size == y.size), "Check dimensions of input matrices"
        assert(y.size >= self.n_bins), "Number of bins should be less than the number of calibration points"
        
        ### All required (hyper-)parameters have been passed correctly
        ### Uniform-mass binning/histogram binning code starts below

        # delta-randomization
        y_score = nudge(y_score, self.delta)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges = get_uniform_mass_bins(y_score, self.n_bins)

        # assign calibration data to bins
        bin_assignment = bin_points(y_score, self.bin_upper_edges)

        # compute bias of each bin 
        self.num_calibration_examples_in_bin = np.zeros([self.n_bins, 1])
        self.mean_pred_values = np.empty(self.n_bins)
        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            self.num_calibration_examples_in_bin[i] = sum(bin_idx)

            # nudge performs delta-randomization
            if (sum(bin_idx) > 0):
                self.mean_pred_values[i] = nudge(y[bin_idx].mean(),
                                                 self.delta)
            else:
                self.mean_pred_values[i] = nudge(0.5, self.delta)

        # check that my code is correct
        assert(np.sum(self.num_calibration_examples_in_bin) == y.size)

        # hurray! histogram binning done
        self.fitted = True

    def predict_proba(self, y_score):
        assert(self.fitted is True), "Call HB.fit() first"
        y_score = y_score.squeeze()

        # delta-randomization
        y_score = nudge(y_score, self.delta)
        
        # assign test data to bins
        y_bins = bin_points(y_score, self.bin_upper_edges)
            
        # get calibrated predicted probabilities
        y_pred_proba = self.mean_pred_values[y_bins]
        return y_pred_proba

    def predict(self, y_score):
        return (self.predict_proba(y_score) >= 0.5).astype('int')

    def score(self, y_score, y):
        y_pred = self.predict(y_score)
        return (y_pred == y).mean()

