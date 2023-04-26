import numpy as np
import cvxpy as cp
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from utils import *
import utils 

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

def sigmoid(x):
    return 1 / (1 + np.exp(-np.minimum(np.maximum(x, -1e2), 1e2)))

def fit_platt_scaling_parameters(scores, labels, regularization=False):
    if regularization:
        # compute proportions in the calibration set in case of regularization being applied
        N_1 = (labels == 1).sum()
        N_0 = (labels == 0).sum()
        t = (labels == 1).astype('int') * (N_1 + 1) / (N_1 + 2) + (
            labels == 0).astype('int') * 1.0 / (N_0 + 2)
    else:
        # just use raw labels
        t = np.copy(labels)

    A = cp.Variable(1)
    B = cp.Variable(1)

    # form an objective for maximization
    # objective corresponds to the log-likelihood of the score 1/(1+exp(-(Ax+b))) given the label y
    s = A * scores + B
    objective = t @ (s) - np.ones_like(t) @ cp.logistic(s) # = (y * s) - log(1 + e^s)
    
    # solve the problem
    try:
        problem = cp.Problem(cp.Maximize(objective))
        problem.solve()

        if problem.status != 'optimal':
            print("CVXPY did not converge")
        calibrated_probs = sigmoid(A.value * scores + B.value)
        return calibrated_probs, A.value, B.value
    except:
        print("CVXPY optimization failed, adding regularization and trying again")
        problem = cp.Problem(cp.Maximize(objective - cp.square(A) - cp.square(B)))
        problem.solve()
        if problem.status != 'optimal':
            print("CVXPY did not converge")
        calibrated_probs = sigmoid(A.value * scores + B.value)
        return calibrated_probs, A.value, B.value

def fit_beta_scaling_parameters(scores_1, scores_2, labels, regularization=False):
    if regularization:
        # compute proportions in the calibration set in case of regularization being applied
        N_1 = (labels == 1).sum()
        N_0 = (labels == 0).sum()
        t = (labels == 1).astype('int') * (N_1 + 1) / (N_1 + 2) + (
            labels == 0).astype('int') * 1.0 / (N_0 + 2)
    else:
        # just use raw labels
        t = np.copy(labels)

    A = cp.Variable(1)
    B = cp.Variable(1)
    C = cp.Variable(1)

    # form an objective for maximization
    # objective corresponds to the log-likelihood of the score 1/(1+exp(-(Ax+b))) given the label y
    s = A * scores_1 + B * scores_2 + C
    objective = t @ (s) - np.ones_like(t) @ cp.logistic(s) # = (y * s) - log(1 + e^s)
    
    # solve the problem
    try:
        problem = cp.Problem(cp.Maximize(objective))
        problem.solve()

        if problem.status != 'optimal':
            print("CVXPY did not converge")
        calibrated_probs = sigmoid(A.value * scores_1 + B.value * scores_2 + C.value)
        return calibrated_probs, A.value, B.value, C.value
    except:
        print("CVXPY optimization failed, adding regularization and trying again")
        problem = cp.Problem(cp.Maximize(objective - cp.square(A) - cp.square(B) - cp.square(C)))
        problem.solve()
        if problem.status != 'optimal':
            print("CVXPY did not converge")
        calibrated_probs = sigmoid(A.value * scores_1 + B.value * scores_2 + C.value)
        return calibrated_probs, A.value, B.value, C.value
    
def sigmoid(x):
    return 1 / (1 + np.exp(-np.minimum(np.maximum(x, -1e2), 1e2)))

def online_platt_scaling_newton(preds, y, beta = 0.1, D = 1):
    assert(np.sum(y==0) + np.sum(y==1) == y.size)
    assert(y.size == preds.size)

    def platt_gradient(x, a, b, y):
        if(y == 1):
            gradient = (sigmoid(x*a + b) - 1) * np.array([x, 1])
        else: 
            gradient = (sigmoid(x*a + b)) * np.array([x, 1])
        return gradient
    
    pred_probs_platt = np.zeros((preds.size))

    online_platt_a = np.zeros((preds.size)) 
    online_platt_b = np.zeros((preds.size))
    pred_probs_platt[0] = sigmoid(online_platt_a[0]*preds[0] + online_platt_b[0])

    norm_control = 1e2
    cum_hessian = (1/(beta*D)**2) * np.eye(2)
    cum_hessian_inv = (beta*D)**2 * np.eye(2)
    
    for i in range(0, preds.size-1):
        g_i = platt_gradient(preds[i], online_platt_a[i], online_platt_b[i], y[i])
        cum_hessian_inv = cum_hessian_inv - np.outer(cum_hessian_inv @ g_i, g_i.T @ cum_hessian_inv)/(1 + (g_i.T @ cum_hessian_inv @ g_i))
        
        cum_hessian = cum_hessian + np.outer(g_i, g_i)
        update = -(1/beta) * (cum_hessian_inv @ g_i)
        
        online_platt_a[i+1] = online_platt_a[i] + update[0]
        online_platt_b[i+1] = online_platt_b[i] + update[1]
        
        if((online_platt_a[i+1]**2 + online_platt_b[i+1]**2) > norm_control**2):
            x = cp.Variable(2)
            x.value = [online_platt_a[i+1], online_platt_b[i+1]]
            linear_factor = -2 * (cum_hessian/i) @ np.array([online_platt_a[i+1], online_platt_b[i+1]])
            constraint = [cp.norm(x)<=norm_control]            
            prob = cp.Problem(cp.Minimize(linear_factor.T @ x + cp.quad_form(x, (cum_hessian/i))), constraint)
            try:
                prob.solve(warm_start=True)
                online_platt_a[i+1] = x.value[0]
                online_platt_b[i+1] = x.value[1]
            except:
                online_platt_a[i+1] = online_platt_a[i]
                online_platt_b[i+1] = online_platt_b[i]
        
        pred_probs_platt[i+1] = sigmoid(online_platt_a[i+1]*preds[i+1] + online_platt_b[i+1])

    return pred_probs_platt, online_platt_a, online_platt_b

def online_beta_scaling_newton(preds_1, preds_2, y, beta = 0.1, D = 2):
    assert(np.sum(y==0) + np.sum(y==1) == y.size)
    assert(y.size == preds_1.size)

    def beta_gradient(x_1, x_2, a, b, c, y):
        if(y == 1):
            gradient = (sigmoid(x_1*a + x_2*b + c) - 1) * np.array([x_1, x_2, 1])
        else: 
            gradient = (sigmoid(x_1*a + x_2*b + c)) * np.array([x_1, x_2, 1])
        return gradient
    
    pred_probs_platt = np.zeros((preds_1.size))

    online_platt_a = np.zeros((preds_1.size)) 
    online_platt_b = np.zeros((preds_1.size))
    online_platt_c = np.zeros((preds_1.size))
    pred_probs_platt[0] = sigmoid(online_platt_a[0]*preds_1[0] + online_platt_b[0]*preds_2[0] + online_platt_c[0])

    norm_control = 1e2
    cum_hessian = (1/(beta*D)**2) * np.eye(3)
    cum_hessian_inv = (beta*D)**2 * np.eye(3)
    
    for i in range(0, preds_1.size-1):
        g_i = beta_gradient(preds_1[i], preds_2[i], online_platt_a[i], online_platt_b[i], online_platt_c[i], y[i])
        cum_hessian_inv = cum_hessian_inv - np.outer(cum_hessian_inv @ g_i, g_i.T @ cum_hessian_inv)/(1 + (g_i.T @ cum_hessian_inv @ g_i))
        
        cum_hessian = cum_hessian + np.outer(g_i, g_i)
        update = -(1/beta) * (cum_hessian_inv @ g_i)
        
        online_platt_a[i+1] = online_platt_a[i] + update[0]
        online_platt_b[i+1] = online_platt_b[i] + update[1]
        online_platt_c[i+1] = online_platt_c[i] + update[2]
        
        if((online_platt_a[i+1]**2 + online_platt_b[i+1]**2 + online_platt_c[i+1]**2) > norm_control**2):
            x = cp.Variable(3)
            x.value = [online_platt_a[i+1], online_platt_b[i+1], online_platt_c[i+1]]
            linear_factor = -2 * (cum_hessian/i) @ np.array([online_platt_a[i+1], online_platt_b[i+1]], online_platt_c[i+1])
            constraint = [cp.norm(x)<=norm_control]            
            prob = cp.Problem(cp.Minimize(linear_factor.T @ x + cp.quad_form(x, (cum_hessian/i))), constraint)
            try:
                prob.solve(warm_start=True)
                online_platt_a[i+1] = x.value[0]
                online_platt_b[i+1] = x.value[1]
                online_platt_c[i+1] = x.value[2]
            except:
                online_platt_a[i+1] = online_platt_a[i]
                online_platt_b[i+1] = online_platt_b[i]
                online_platt_c[i+1] = online_platt_c[i]
        
        pred_probs_platt[i+1] = sigmoid(online_platt_a[i+1]*preds_1[i+1] + online_platt_b[i+1]*preds_2[i+1] + online_platt_c[i+1])

    return pred_probs_platt, online_platt_a, online_platt_b, online_platt_c


## implements online logistic regression from http://proceedings.mlr.press/v125/jezequel20a/jezequel20a.pdf
## eta_factor = 1/1(1+BR); see eqn (2) in the paper
## lambda is the regularization parameter; see eqn (3) in the paper

def online_platt_scaling_aioli(preds, y, eta_factor = 0.1, reg_lambda = 1):
    # preds and y should both one-dimensional numpy arrays of the same size
    # Example: if data is x_test, y_test and model is rf, call like the following
    # pred_probs_test = rf.predict_proba(x_test)[:,1]
    # pred_probs_calibrated = online_platt_scaling(pred_probs_test, y_test)

    # assert(np.min(preds) >= 0 and np.max(preds <= 1))
    assert(False), "Call online_platt_scaling_newton instead"
    
    assert(np.sum(y==0) + np.sum(y==1) == y.size)
    assert(y.size == preds.size)

    def platt_gradient(x, a, b, y):
        if(y == 1):
            gradient = (sigmoid(x*a + b) - 1) * np.array([x, 1])
        else: 
            gradient = (sigmoid(x*a + b)) * np.array([x, 1])
        return gradient

    pred_probs_platt = np.zeros((preds.size))
    preds_before_logistic = np.zeros((preds.size))
    
    online_platt_a = np.zeros((preds.size))
    online_platt_b = np.zeros((preds.size))
    g_s = np.zeros((preds.size, 2))

    warm_start = 0
    accumulate_terms = 0

    linear_factor = np.zeros((2))
    quadratic_factor = reg_lambda * np.identity(2)
    x = cp.Variable(2)
    x.value = [0,0]

    for i in tqdm(range(preds.size), disable=True):
        if(i > accumulate_terms):
            s = i-1
            eta_s = eta_factor * np.exp(2*(y[s]-0.5)*preds_before_logistic[s])
            
            linear_factor +=  (1 - 
                               eta_s * np.dot(g_s[s,:], np.array(online_platt_a[s], online_platt_b[s]))
                               ) * g_s[s,:]
            quadratic_factor += 0.5*eta_s*np.outer(g_s[s,:], g_s[s,:])
    
            x.value = [online_platt_a[s], online_platt_b[s]]

        if(i >= warm_start):
            preds_with_constant = np.array([preds[i], 1])
            prob = cp.Problem(cp.Minimize(linear_factor.T @ x + 
                                          cp.quad_form(x, quadratic_factor)
                                          + cp.logistic(-1 * (preds_with_constant @ x))
                                          + cp.logistic(1 * (preds_with_constant @ x))
                                          )
                              )
            
            # prob.solve(warm_start=True)
    
            try:
                prob.solve(warm_start=True)
            except:
                x.value = [online_platt_a[i-1], online_platt_b[i-1]]
                # print(x.value)
                # print(linear_factor)
                # print(quadratic_factor)
                # print(preds_with_constant)
            
            online_platt_a[i] = x.value[0]
            online_platt_b[i] = x.value[1]

        else:
            online_platt_a[i] = fixed_platt_a
            online_platt_b[i] = fixed_platt_b
    
        preds_before_logistic[i] = online_platt_a[i]*preds[i] + online_platt_b[i]
        pred_probs_platt[i] = sigmoid(preds_before_logistic[i])

        g_s[i,:] = platt_gradient(preds[i], online_platt_a[i], online_platt_b[i], y[i])

    return pred_probs_platt, online_platt_a, online_platt_b
