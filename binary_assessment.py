import numpy as np
from utils import *

def reliability_diagram(y, pred_proba, ax, color=None, n_bins=15):
    if(len(np.unique(pred_proba))
           <= (pred_proba.shape[0]/10)):
        print("Classifier has discrete output. Further binning not done for plotting reliability diagram.")
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_proba)
    else:
        print("Using {:d} adaptive bins for plotting reliability diagram.".format(n_bins))
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_continuous(y, pred_proba, n_bins)

    if(color != None):
        ax.scatter(pi_pred, pi_true, color=color)
    else:
        ax.scatter(pi_pred, pi_true)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")        

    ax.plot([0,1],[0,1],'k--',alpha=0.7)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    
def sharpness(y, pred_proba, n_bins=15):
    if(len(np.unique(pred_proba))
       <= (pred_proba.shape[0]/10)):
        print("Classifier has discrete output. Further binning not done for sharpness estimation.")
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_proba)
    else:
        print("Using {:d} adaptive bins for sharpness estimation.".format(n_bins))        
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_continuous(y, pred_proba, n_bins)

    assert(sum(n_elem) == y.size)
    return np.sum(n_elem * (pi_true**2))/y.size        

def ece(y, pred_proba, n_bins=15):
    if(len(np.unique(pred_proba))
       <= (pred_proba.shape[0]/10)):
        print("Classifier has discrete output. Further binning not done for ECE estimation.")
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_proba)
    else:
        print("Using {:d} adaptive bins for ECE estimation.".format(n_bins))        
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_continuous(y, pred_proba, n_bins)
    assert(sum(n_elem) == y.size)
    return np.sum(n_elem * np.abs(pi_pred - pi_true))/y.size


def get_binned_probabilities_discrete(y, pred_proba, pred_proba_base = None):
    assert(len(np.unique(pred_proba))
           <= (pred_proba.shape[0]/10)), "Predicted probabilities are not sufficiently discrete; using corresponding continuous method"
    bin_edges = np.sort(np.unique(pred_proba))
    true_n_bins = len(bin_edges)
    pi_pred = np.zeros(true_n_bins)
    pi_base = np.zeros(true_n_bins)
    pi_true = np.zeros(true_n_bins)
    n_elem = np.zeros(true_n_bins)
    bin_assignment = bin_points(pred_proba, bin_edges)

    for i in range(true_n_bins):
        bin_idx = (bin_assignment == i)
        assert(sum(bin_idx) > 0), "This assert should pass by construction of the code"
        n_elem[i] = sum(bin_idx)
        pi_pred[i] = pred_proba[bin_idx].mean()
        if(pred_proba_base is not None):
            pi_base[i] = pred_proba_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()    

    assert(sum(n_elem) == y.size)

    return n_elem, pi_pred, pi_base, pi_true

def get_binned_probabilities_continuous(y, pred_proba, n_bins, pred_proba_base = None):
    pi_pred = np.zeros(n_bins)
    pi_base = np.zeros(n_bins)
    pi_true = np.zeros(n_bins)
    n_elem = np.zeros(n_bins)
    bin_assignment = bin_points_uniform(pred_proba, n_bins)
    
    for i in range(n_bins):
        bin_idx = (bin_assignment == i)
        assert(sum(bin_idx) > 0), "This assert should pass by construction of the code"
        n_elem[i] = sum(bin_idx)
        pi_pred[i] = pred_proba[bin_idx].mean()
        if(pred_proba_base is not None):
            pi_base[i] = pred_proba_base[bin_idx].mean()
        pi_true[i] = y[bin_idx].mean()    

    assert(sum(n_elem) == y.size)

    return n_elem, pi_pred, pi_base, pi_true

def validity_plot(y, pred_proba, ax, color=None, n_bins=15):
    if(len(np.unique(pred_proba))
           <= (pred_proba.shape[0]/10)):
        print("Classifier has discrete output. Further binning not done for making validity plot.")
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_proba)
    else:
        print("Using {:d} adaptive bins for making validity plot.".format(n_bins))
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_continuous(y, pred_proba, n_bins)

    Delta = np.abs(pi_pred - pi_true)
    validity_plot_delta(Delta, n_elem, ax, color)

def conditional_validity_plot(y, pred_proba, ax, color=None, n_bins=15):
    if(len(np.unique(pred_proba))
           <= (pred_proba.shape[0]/10)):
        print("Classifier has discrete output. Further binning not done for making conditional validity plot.")
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_discrete(y, pred_proba)
    else:
        print("Using {:d} adaptive bins for making conditional validity plot.".format(n_bins))
        n_elem, pi_pred, _, pi_true = get_binned_probabilities_continuous(y, pred_proba, n_bins)

    Delta = np.abs(pi_pred - pi_true)
    conditional_validity_plot_delta(Delta, n_elem, ax, color)


def validity_plot_delta(Delta, n_elem, ax, color=None):
    assert(np.shape(Delta) == np.shape(n_elem))
    assert(np.size(Delta) == np.shape(Delta)[0]), "this function makes a validity plot for a single run, use function validity_plot_aggregate for multiple runs"
    if(np.shape(np.shape(Delta))[0] == 1):
        Delta = np.expand_dims(Delta, axis=0)
        n_elem = np.expand_dims(n_elem, axis=0)        
    n_points = sum(n_elem[0,:])

    cdf = lambda x: np.diag((Delta <= x) @ n_elem.T)/n_points

    dx = 0.001
    xs  = np.arange(0, 1.0, dx)
    xmaxind = xs.size - 1
    ys = np.zeros(xs.shape)
    for i in range(xs.size):
        ys[i] = cdf(xs[i])
        if(ys[i] == 1.0):
            xmaxind = i
            break
    ys[xmaxind:] = 1.0
    if(color != None):
        handle = ax.plot(xs, ys, color=color)
    else:
        handle = ax.plot(xs, ys)

    ax.set_xlim([0, min(xs[xmaxind] + 500*dx, 1.0)])
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$V(\epsilon)$')
    ax.grid('on')
    return handle[0]
    
def validity_plot_aggregate(Delta, n_elem, ax, color=None):
    assert(np.shape(Delta) == np.shape(n_elem))
    assert(np.size(Delta) > np.shape(Delta)[0]), "this function makes a validity plot for multiple runs, use function validity_plot for a single run"
    if(np.shape(np.shape(Delta))[0] == 1):
        Delta = np.expand_dims(Delta, axis=0)
        n_elem = np.expand_dims(n_elem, axis=0)        
    n_sims = n_elem.shape[0]
    n_points = sum(n_elem[0,:])

    cdf = lambda x: np.diag((Delta <= x) @ n_elem.T)/n_points

    dx = 0.001
    xs  = np.arange(0, 1.0, dx)
    ys = np.array([np.mean(cdf(x)) for x in xs])
    yerrors = np.array([np.std(cdf(x))/np.sqrt(n_sims) for x in xs])

    if(color != None):    
        handle = ax.errorbar(xs, ys, yerr=yerrors, color=color)
    else:
        handle = ax.errorbar(xs, ys, yerr=yerrors)


    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$V(\epsilon)$')
    return handle[0]

def conditional_validity_plot_delta(Delta, n_elem, ax, color=None):
    assert(np.shape(Delta) == np.shape(n_elem))
    if(np.shape(np.shape(Delta))[0] == 1):
        Delta = np.expand_dims(Delta, axis=0)
        n_elem = np.expand_dims(n_elem, axis=0)        
    n_sims = n_elem.shape[0]
    n_points = sum(n_elem[0,:])
    
    # Assuming that Delta = 0 whenever n_elem = 0
    cdf = lambda x: np.min(Delta <= x, axis=1)
    dx = 0.001
    xs  = np.arange(0, 1.0, dx)
    ys = np.array([np.mean(cdf(x)) for x in xs])
    yerrors = np.array([np.std(cdf(x))/np.sqrt(n_sims) for x in xs])

    if(color != None):    
        handle = ax.errorbar(xs, ys, yerr=yerrors, color=color)
    else:
        handle = ax.errorbar(xs, ys, yerr=yerrors)

    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$V(\epsilon)$')
    return handle[0]

# Following code was used internally for experiments with canonical calibration.
# I have not cleaned, tested, or user-interfaced it, but it should be usable with some effort.
# Please contact me (https://aigen.github.io) if you have trouble.
def plot_calibration_figures(X, y, y_recal,
                             clf, recalibrated_clf,
                             n_bins,
                             fig, ax, title_str, show_legend,
                             color_clf = None, color_recal = None,
                             points_per_bin = False):
    pred_proba_base = clf(X)
    if len(pred_proba_base.shape) > 1:
        pred_proba_base = pred_proba_base[:, 1]
    pred_proba = recalibrated_clf(X)

    if(len(np.unique(pred_proba))
       <= (pred_proba.shape[0]/10)):
        n_elem, pi_calibrated, pi_base, pi_true =\
            get_binned_probabilities_discrete(y, pred_proba, pred_proba_base)
        assert(np.sum(n_elem) == X.shape[0])
        
        pi_true_uncalibrated = pi_true

        base_ece = np.sum(n_elem * np.abs(pi_base - pi_true))/np.sum(n_elem)
        hist_ece = np.sum(n_elem * np.abs(pi_calibrated - pi_true_uncalibrated))/np.sum(n_elem)
        sharpness = np.sum(n_elem * (pi_true**2))/np.sum(n_elem)
    else:
        n_elem, pi_calibrated, _, pi_true =\
            get_binned_probabilities_continuous(y_recal, pred_proba, n_bins)
        n_elem, pi_base, _, pi_true_uncalibrated =\
            get_binned_probabilities_continuous(y, pred_proba_base, n_bins)
        
        base_ece = np.sum(n_elem * np.abs(pi_base - pi_true_uncalibrated))/np.sum(n_elem)
        hist_ece = np.sum(n_elem * np.abs(pi_calibrated - pi_true))/np.sum(n_elem)
        sharpness = -1
        
    if(color_clf != None):
        handle0 = validity_plot(np.abs(pi_base - pi_true_uncalibrated),
                                n_elem, ax[1], color_clf)
        lns1 = ax[0].scatter(pi_base, pi_true_uncalibrated, label="Base", color=color_clf)
    else:
        lns1 = ax[0].scatter(pi_base, pi_true_uncalibrated, label="Base")
        handle0 = validity_plot(np.abs(pi_base - pi_true_uncalibrated), n_elem, ax[1])
    if(color_recal != None):
        lns2 = ax[0].scatter(pi_calibrated, pi_true, label="Recalibrated", color=color_recal)
        handle1 = validity_plot(np.abs(pi_calibrated - pi_true),
                                n_elem, ax[1], color_recal)                            
    else:
        lns2 = ax[0].scatter(pi_calibrated, pi_true, label="Recalibrated")
        handle1 = validity_plot(np.abs(pi_calibrated - pi_true), n_elem, ax[1])
                            
    ax[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", alpha=0.7)
    ax[0].set_xlabel("Predicted probability")
    ax[0].set_ylabel("True probability")
    ax[1].set_xlim((0, 0.16))

    ax[0].grid(True, linestyle='--')
    ax[1].grid(True, linestyle='--')    

    if(points_per_bin==True):
        ax0_right = ax[0].twinx()
        lns3 = ax0_right.scatter(pi_calibrated, n_elem,
                                 marker="+", linewidths = 1.0,
                                 alpha=0.8, s=50, c='black',
                                 label='#points')
        ax0_right.set_ylabel("#points with predicted value")

        n_elem_min = 5*np.floor(np.min(n_elem)/5)
        n_elem_max = 5*np.ceil(np.max(n_elem)/5)
        mi_twin = np.floor(n_elem_min - 0.05*(n_elem_max - n_elem_min))
        ma_twin = np.ceil(n_elem_max + 0.05*(n_elem_max - n_elem_min))
        ax0_right.set_ylim([mi_twin, ma_twin])
        ax0_right.set_yticks(np.linspace(n_elem_min, n_elem_max, 6))
        ax0_right.grid(None)
        lns = [lns1, lns2, lns3]
    else:
        lns = [lns1, lns2]        
    
    labs = [l.get_label() for l in lns]

    if(show_legend):
        ax[1].legend([handle0, handle1],
                     ["Base", "Recalibrated"],
                     loc=(ax[0].get_position().x1 + 2,
                          ax[0].get_position().y1 - 2))
        ax[0].legend(lns, labs,
                     loc=(ax[0].get_position().x1 + 2,
                          ax[0].get_position().y0))

        ax[0].set_title("Class {} reliability diagram".format(title_str))
        ax[1].set_title("Class {} validity plot".format(title_str))

    return base_ece, hist_ece, sharpness 
    
