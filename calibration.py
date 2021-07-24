import numpy as np
from utils import *

class HB_binary(object):
    def __init__(self, n_bins=15):
        ### Hyperparameters
        self.delta = 1e-10
        self.n_bins = n_bins

        ### Parameters to be learnt 
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None

        ### Internal variables
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

        # histogram binning done
        self.fitted = True

    def predict_proba(self, y_score):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        y_score = y_score.squeeze()

        # delta-randomization
        y_score = nudge(y_score, self.delta)
        
        # assign test data to bins
        y_bins = bin_points(y_score, self.bin_upper_edges)
            
        # get calibrated predicted probabilities
        y_pred_prob = self.mean_pred_values[y_bins]
        return y_pred_prob

class HB_toplabel(object):
    def __init__(self, points_per_bin=50):
        ### Hyperparameters
        self.points_per_bin = points_per_bin

        ### Parameters to be learnt 
        self.hb_binary_list = []
        
        ### Internal variables
        self.num_classes = None
    
    def fit(self, pred_mat, y):
        assert(self.points_per_bin is not None), "Points per bins has to be specified"
        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        y = y.squeeze()
        assert(pred_mat.shape[0] == y.size), "Check dimensions of input matrices"
        self.num_classes = pred_mat.shape[1]
        assert(np.min(y) >= 1 and np.max(y) <= self.num_classes), "Labels should be numbered 1 ... L, where L is the number of columns in the prediction matrix"
        
        top_score = np.max(pred_mat, axis=1).squeeze()
        pred_class = (np.argmax(pred_mat, axis=1)+1).squeeze()

        for l in range(1, self.num_classes+1, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)

            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}.".format(l, self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict_proba(self, pred_mat):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        assert(self.num_classes == pred_mat.shape[1]), "Number of columns of prediction matrix do not match number of labels"
        
        top_score = np.max(pred_mat, axis=1).squeeze()
        pred_class = (np.argmax(pred_mat, axis=1)+1).squeeze()

        n = pred_class.size
        pred_top_score = np.zeros((n))
        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]-1].predict_proba(top_score[i])

        return pred_top_score

    def fit_top(self, top_score, pred_class, y):
        assert(self.points_per_bin is not None), "Points per bins has to be specified"

        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        y = y.squeeze()

        assert(min(np.min(y), np.min(pred_class)) >= 1), "Labels should be numbered 1 ... L, use HB_binary for a binary problem"
        assert(top_score.size == y.size), "Check dimensions of input matrices"
        assert(pred_class.size == y.size), "Check dimensions of input matrices"
        assert(y.size >= self.n_bins), "Number of bins should be less than the number of calibration points"

        self.num_classes = max(np.max(y), np.max(pred_class))
        
        for l in range(1, self.num_classes+1, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)

            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}".format(self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict_proba_top(self, top_score, pred_class):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        assert(top_score.size == pred_class.size), "Check dimensions of input matrices"
        assert(np.min(pred_class) >= 1 and np.min(pred_class) <= self.num_classes), "Some of the predicted labels are not in the range of labels seen while calibrating"
        n = pred_class.size
        pred_top_score = np.zeros((n))
        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]-1].predict_proba(top_score[i])

        return pred_top_score
        
