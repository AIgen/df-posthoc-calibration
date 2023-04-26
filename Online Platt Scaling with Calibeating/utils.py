import numpy as np
import copy as copy
import assessment

def get_uniform_mass_bins(probs, n_bins):
    assert(probs.size >= n_bins), "Fewer points than bins"
    
    probs_sorted = np.sort(probs)

    # split probabilities into groups of approx equal size
    groups = np.array_split(probs_sorted, n_bins)
    bin_edges = list()
    bin_upper_edges = list()

    for cur_group in range(n_bins-1):
        bin_upper_edges += [max(groups[cur_group])]
    bin_upper_edges += [np.inf]

    return np.array(bin_upper_edges)

def digitize_predictions(scores, n_bins):
    bin_rightedges = np.linspace(1.0/n_bins, 1.0, n_bins)
    scores_digitized = bin_rightedges[bin_points(scores, bin_rightedges)] - 0.5/n_bins
    assert(np.all(np.abs(scores_digitized - scores) <= 1e-5 + (0.5/n_bins)))
    return scores_digitized

def digitize_predictions_index(scores, n_bins):
    bin_rightedges = np.linspace(1.0/n_bins, 1.0, n_bins)
    digitization_indices = bin_points(scores, bin_rightedges)
    scores_digitized = bin_rightedges[digitization_indices] - 0.5/n_bins
    assert(np.all(np.abs(scores_digitized - scores) <= 1e-5 + (0.5/n_bins)))
    return digitization_indices

def bin_points(scores, bin_edges):
    assert(bin_edges is not None), "Bins have not been defined"
    scores = scores.squeeze()
    assert(scores.max() <= 1), "Maximum score value is > 1!"
    assert(np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)

def bin_points_uniform(x, n_bins):
    x = nudge(x.squeeze())
    bin_upper_edges = get_uniform_mass_bins(x, n_bins)
    return np.sum(x.reshape((-1, 1)) > bin_upper_edges, axis=1)

def nudge(matrix, delta=1e-10):
    return((matrix + np.random.uniform(low=0,
                                       high=delta,
                                       size=(matrix.shape)))/(1+delta))

class identity():
    def predict_proba(self, x):
        return x
    def predict(self, x):
        return np.argmax(x, axis=1)


def randomized_game(sequence, m, pivot):
    # implements Foster [1999]
    ###### set up some constants
    eps = 1/(2*m)
    assert(np.round(1/(2*eps)) == m)
    L = np.arange(0, 1, 2*eps)
    R = L + 2*eps
    M = L + eps
    ######

    T = sequence.size
    if(T==1):
        return 0
    N = np.zeros((m))
    p = copy.copy(M)
    
    d = L - p 
    e = p - R
    preds = np.zeros((T))
    
    for t in range(T):
        if((d[pivot] <= 0) * (e[pivot] <= 0) > 0):
            i = pivot
        elif np.sum((d <= 0) * (e <= 0)) > 0:
            i = np.where((d <= 0) * (e <= 0))[0][0]
        else:
            i = np.where(d > 0)[0][0] - 1
            assert(d[i+1] > 0 and e[i] > 0)
            play = (i, i+1)
            i = play[0 if (np.random.uniform(0,d[i+1]+e[i]) <= d[i+1]) else 1]

        if(t==0):
            assert(i == pivot)
        preds[t] = M[i]
        p[i] = (p[i]*N[i] + sequence[t])/(N[i] + 1)
        N[i] = N[i] + 1
        d[i] = L[i] - p[i] 
        e[i] = p[i] - R[i]
        
    return preds

def plot_all(fig, ax, y_test, N_calib_points,
             pred_probs_base,
             pred_probs_adaptive_platt,
             pred_probs_fixed_platt,
             pred_probs_platt,
             pred_probs_platt_beat,
             pred_probs_platt_calibeat):
    # N_calib_points = np.linspace(200,900,15).astype('int')
    print(N_calib_points)
    ECE_base = []
    ECE_recalib_platt_const = []
    ECE_recalib_platt_adaptive = []
    ECE_recalib_platt_continuous = []
    ECE_recalib_platt_continuous_beat = []
    ECE_recalib_platt_continuous_calibeat = []

    for n_calib_points in N_calib_points:
        ECE_base = ECE_base + [assessment.ece(
            y_test[:n_calib_points], 
            pred_probs_base[:n_calib_points], fixed=True, n_bins=10)]
        
        ECE_recalib_platt_adaptive = ECE_recalib_platt_adaptive + [assessment.ece(
            y_test[:n_calib_points], pred_probs_adaptive_platt[:n_calib_points], fixed=True, n_bins=10)]
    
        ECE_recalib_platt_const = ECE_recalib_platt_const + [assessment.ece(
            y_test[:n_calib_points], pred_probs_fixed_platt[:n_calib_points], fixed=True, n_bins=10)]
        
        ECE_recalib_platt_continuous = ECE_recalib_platt_continuous + [assessment.ece(
            y_test[:n_calib_points], 
            pred_probs_platt[:n_calib_points], fixed=True, n_bins=10)]
    
        ECE_recalib_platt_continuous_beat = ECE_recalib_platt_continuous_beat + [assessment.ece(
            y_test[:n_calib_points], 
            pred_probs_platt_beat[:n_calib_points], fixed=True, n_bins=10)]
    
        ECE_recalib_platt_continuous_calibeat = ECE_recalib_platt_continuous_calibeat + [assessment.ece(
            y_test[:n_calib_points], 
            pred_probs_platt_calibeat[:n_calib_points], fixed=True, n_bins=10)]

    ax.plot(N_calib_points, ECE_base, '-o', 
         label="ECE with no recalibration")
    ax.plot(N_calib_points, ECE_recalib_platt_adaptive, '-o', 
             label="ECE with adaptive Platt recalibration")
    ax.plot(N_calib_points, ECE_recalib_platt_const, '-o', 
             label="ECE with fixed Platt recalibration")
    ax.plot(N_calib_points, ECE_recalib_platt_continuous, '-o', 
             label="ECE with continuous Platt recalibration")
    ax.plot(N_calib_points, ECE_recalib_platt_continuous_beat, '-o', 
             label="ECE with continuous Platt recalibration and beating")
    ax.plot(N_calib_points, ECE_recalib_platt_continuous_calibeat, '-o', 
             label="ECE with continuous Platt recalibration and calibeating")
    
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0.25))
    ax.set_title('Change in ECE with time')

    
def sigmoid(x):
    return 1 / (1 + np.exp(-np.minimum(np.maximum(x, -1e2), 1e2)))

def logit(arr):
    threshold = 1e-5
    arr = np.minimum(np.maximum(arr, threshold), 1-threshold)
    return np.log(arr/(1-arr))

def log_thresholded(arr):
    threshold = 1e-5
    arr = np.minimum(np.maximum(arr, threshold), 1-threshold)
    return np.log(arr)

def normalized_hist(vals, ax):
    plotted_y, plotted_x = np.histogram(vals)
    ax.hist(vals, weights=[1/plotted_y.max()]*vals.size,
            alpha=0.2, label=r'Histogram of $X_t$ values in time interval', color='k')
