import numpy as np
from utils import *
import utils 
import warnings
warnings.filterwarnings("error")

def sharpness(y, pred_prob, n_bins=15, quiet=False, fixed=False):
    if(fixed):
        n_elem, _, _, pi_true = get_binned_probabilities_fixed_width(y, pred_prob, n_bins)
    elif(len(np.unique(pred_prob))
       <= (pred_prob.shape[0]/10)):
        if(not quiet):
            print("Classifier has discrete output. Further binning not done for sharpness estimation.")
        n_elem, _, _, pi_true = get_binned_probabilities_discrete(y, pred_prob)
    else:
        if(not quiet):
            print("Using {:d} adaptive bins for sharpness estimation.".format(n_binsi))        
        n_elem, _, _, pi_true = get_binned_probabilities_continuous(y, pred_prob, n_bins)    
    
    assert(sum(n_elem) == y.size)

    estimate = (n_elem @ (pi_true**2))/y.size
    return estimate 

def ece(y, pred_prob, n_bins=15, quiet=False, fixed=False, l2=False, use_midpoint=False):
    # assert(fixed and n_bins==10 and use_midpoint==False and l2==False)    
    if(fixed):
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_fixed_width(y, pred_prob, n_bins)
    elif(len(np.unique(pred_prob))
       <= (pred_prob.shape[0]/50)):
        if(not quiet):
            print("Classifier has discrete output. Further binning not done for ECE estimation.")
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_prob)
    else:
        if(not quiet):
            print("Using {:d} adaptive bins for ECE estimation.".format(n_bins))
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_continuous(y, pred_prob, n_bins)
    assert(sum(n_elem) == y.size)

    if(fixed and use_midpoint):
        pi_pred = digitize_predictions(pi_pred, n_bins)
    if(l2):
        
        # using debiased estimate
        biased_estimate = np.dot(n_elem, np.power(pi_pred - pi_true, 2))/y.size
        correction = np.dot(n_elem, np.divide(np.multiply(pi_true, 1 - pi_true), np.maximum(n_elem-1, 1)))/y.size
        return biased_estimate - correction
    else:
        return np.dot(n_elem, np.abs(pi_pred - pi_true))/y.size
    
def get_binned_probabilities_fixed_width(y, pred_prob, n_bins, pred_prob_base = None):
    assert(n_bins >= 0)
    bin_edges = np.linspace(1.0/n_bins, 1.0, n_bins)
    pi_pred = np.zeros(n_bins)
    pi_base = np.zeros(n_bins)
    pi_true = np.zeros(n_bins)
    n_elem = np.zeros(n_bins)
    bin_assignment = utils.bin_points(pred_prob, bin_edges)
    
    for i in range(n_bins):
        bin_idx = (bin_assignment == i)
        n_elem[i] = sum(bin_idx)
        if(n_elem[i] == 0):
            continue
        pi_pred[i] = pred_prob[bin_idx].mean()
        if(pred_prob_base is not None):
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()    

    assert(sum(n_elem) == y.size)

    return n_elem, pi_pred, pi_base, pi_true

def get_binned_probabilities_continuous(y, pred_prob, n_bins, pred_prob_base = None):
    pi_pred = np.zeros(n_bins)
    pi_base = np.zeros(n_bins)
    pi_true = np.zeros(n_bins)
    n_elem = np.zeros(n_bins)
    bin_assignment = utils.bin_points_uniform(pred_prob, n_bins)
    
    for i in range(n_bins):
        bin_idx = (bin_assignment == i)
        assert(sum(bin_idx) > 0), "This assert should pass by construction of the code"
        n_elem[i] = sum(bin_idx)
        pi_pred[i] = pred_prob[bin_idx].mean()
        if(pred_prob_base is not None):
            pi_base[i] = pred_prob_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()    

    assert(sum(n_elem) == y.size)

    return n_elem, pi_pred, pi_base, pi_true
