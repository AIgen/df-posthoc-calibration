import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import copy

import assessment 
import calibration 
import utils 
import warnings 

from matplotlib import pyplot as plt 
warnings.filterwarnings("default")

def test_and_update(storage_list, function, y, pred_probs):
    storage_list.append(function(y, pred_probs, fixed=True, n_bins=10))

def churn_order_data_by_location(df):
    df["location"] = 0
    df["noisy_loc"] = 0
    df.loc[df["Spain"] == 1, "location"] = 1
    df.loc[df["Germany"] == 1, "location"] = 2
    df["noisy_loc"] = df["location"]
    df.sort_values(by="noisy_loc", inplace=True)
    n_france = (df["location"] == 0).sum()
    n_spain = (df["location"] == 1).sum()
    n_germany = (df["location"] == 2).sum()
    assert(df.shape[0] == n_france + n_germany + n_spain)
    
    X = df.drop(columns=["noisy_loc", "location", "Spain", "Germany", "Exited"]).to_numpy()
    Y = df["Exited"]
    Y = Y.to_numpy()
    
    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]

    return x_train, y_train, x_test, y_test

def churn_order_data_by_age(df):
    shuffle_level = 1
    df['Noisy_age'] = df['Age'] + np.random.randint(-shuffle_level, shuffle_level+1, (df.shape[0]))
    df = df.sort_values(by ='Noisy_age')
    
    X = df.drop(columns=['Exited', 'Noisy_age']).to_numpy()
    Y = df['Exited']
    Y = Y.to_numpy()
    
    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]

    return x_train, y_train, x_test, y_test

def churn_order_data_by_tenure(df):
    df = df.sort_values(by ='Tenure')
    X = df.drop(columns=['Exited']).to_numpy()
    Y = df['Exited']
    Y = Y.to_numpy()

    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]

    return x_train, y_train, x_test, y_test

def churn_iid(df):
    X = df.drop(columns=['Exited']).to_numpy()
    Y = df['Exited']
    Y = Y.to_numpy()

    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]
    
    return x_train, y_train, x_test, y_test

def credit_iid(df):
    X = df.drop(columns=['default payment next month']).to_numpy()
    Y = df['default payment next month']
    Y = Y.to_numpy()

    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]
    
    return x_train, y_train, x_test, y_test

def credit_order_by_age(df):
    shuffle_level = 1
    df['noisy_age'] = df['AGE'] + np.random.randint(-shuffle_level, shuffle_level+1, (df.shape[0]))
    df = df.sort_values(by ='noisy_age')

    X = df.drop(columns=['noisy_age', 'default payment next month']).to_numpy()
    Y = df['default payment next month']
    Y = Y.to_numpy()
    
    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]
    
    return x_train, y_train, x_test, y_test

def credit_order_by_sex(df):
    df['noisy_sex'] = df['SEX'] + np.random.normal(0, 0.05, (df.shape[0]))
    df = df.sort_values(by ='noisy_sex')

    X = df.drop(columns=['noisy_sex', 'default payment next month']).to_numpy()
    Y = df['default payment next month']
    Y = Y.to_numpy()
    
    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]
    
    return x_train, y_train, x_test, y_test

def bank_order_by_age(df):
    shuffle_level = 1
    df['noisy_age'] = df['age'] + np.random.randint(-shuffle_level, shuffle_level+1, (df.shape[0]))
    df = df.sort_values(by ='noisy_age')
    X = df.drop(columns=['y', 'noisy_age']).to_numpy()
    Y = df['y'].to_numpy()

    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]
    
    return x_train, y_train, x_test, y_test

def bank_order_by_balance(df):
    df['noisy_balance'] = df['balance'] + np.random.normal(0, 100, (df.shape[0]))
    df = df.sort_values(by ='noisy_balance')

    X = df.drop(columns=['noisy_balance', 'y']).to_numpy()
    Y = df['y'].to_numpy()

    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]
    
    return x_train, y_train, x_test, y_test

def bank_iid(df):
    X = df.drop(columns=['y']).to_numpy()
    Y = df['y'].to_numpy()

    x_train = X[:1000,:]
    y_train = Y[:1000]
    x_test = X[1000:,:]
    y_test = Y[1000:]
    
    return x_train, y_train, x_test, y_test

def fetal_order_by_acceleration(df):
    df['noisy_acceleration'] = df['accelerations'] + np.random.normal(0, 0.001, (df.shape[0]))
    df = df.sort_values(by ='noisy_acceleration')

    X = df.drop(columns=['noisy_acceleration', 'fetal_health']).to_numpy()
    Y = df['fetal_health']
    Y = Y.to_numpy()

    x_train = X[:626,:]
    y_train = Y[:626]
    x_test = X[626:,:]
    y_test = Y[626:]
    
    return x_train, y_train, x_test, y_test

def fetal_iid(df):
    X = df.drop(columns=['fetal_health']).to_numpy()
    Y = df['fetal_health']
    Y = Y.to_numpy()

    x_train = X[:626,:]
    y_train = Y[:626]
    x_test = X[626:,:]
    y_test = Y[626:]
    
    return x_train, y_train, x_test, y_test

def orth_vec(beta):
    beta_norm = np.linalg.norm(beta)
    beta_hat = beta/beta_norm
    d = beta.size
    v = np.random.randn((d))
    v = v - beta_hat*(v@beta_hat)
    v /= np.linalg.norm(v)
    v *= beta_norm
    return v

def covariate_shift(delta, n_train, n_test):
    assert(n_train == 1000 and n_test == 5000)

    if(delta==0):
        divide_factor = np.inf
    else:
        divide_factor = (n_train+n_test)*(180/delta)
    
    d = 10
    d_cross = int(d + ((d*(d-1))/2))
    d_all = d + d_cross
    
    beta = 2*np.random.randint(0,2,size=d_all)-1
    def create_data(n):
        dat = np.zeros((n, d_all))
        Y = np.zeros((n))
        for i in range(n):
            new_dat = np.random.randn((d))
            vcur = create_data.v1*np.cos(create_data.theta) + create_data.v2*np.sin(create_data.theta)
            new_dat = new_dat + 10*vcur*(new_dat@vcur)
            dat[i,:] = preprocessing.PolynomialFeatures(2, include_bias=False).fit_transform(new_dat.reshape(-1,1).T)

            Y[i] = int(np.random.random() <= utils.sigmoid(dat[i,:]@beta))

            create_data.theta += np.pi/divide_factor
        return dat[:,:d], Y

    create_data.v1 = np.random.randn((d))
    create_data.v1 /= np.linalg.norm(create_data.v1)
    create_data.v2 = orth_vec(create_data.v1)
    create_data.theta = 0

    x_train, y_train = create_data(n_train)
    x_test, y_test = create_data(n_test)
        
    return x_train, y_train, x_test, y_test

def label_shift(final_bias, n_train, n_test):
    initial_bias = 0.5
    delta = (final_bias-initial_bias)/(n_train+n_test)
    d = 10
    def create_data(n):
        Y = np.zeros((n))
        for i in range(n):
            Y[i] = int(np.random.random() <= create_data.bias)
            create_data.bias += delta

        dat = np.zeros((n,d))
        dat[Y==0, :] = np.random.randn(np.sum(Y==0), d)
        dat[Y==1, :] = np.random.randn(np.sum(Y==1), d)

        dat[Y==0, 0] *= 2
        dat[Y==1, 0] += 1
        dat[Y==1, 1] *= 2

        return dat, Y

    create_data.bias = initial_bias
    
    x_train, y_train = create_data(n_train)
    x_test, y_test = create_data(n_test)
        
    return x_train, y_train, x_test, y_test

def regression_shift(delta, n_train, n_test):
    assert(n_train == 1000 and n_test == 5000)

    if(delta==0):
        divide_factor = np.inf
    else:
        divide_factor = (n_train+n_test)*(180/delta)
    
    d = 10
    d_cross = int(d + ((d*(d-1))/2))
    d_all = d + d_cross
    def create_data(n):
        dat = np.random.randn(n, d)
        dat = preprocessing.PolynomialFeatures(2, include_bias=False).fit_transform(dat)
        Y = np.zeros((n))
        for i in range(n):
            beta = create_data.v1*np.cos(create_data.theta) + create_data.v2*np.sin(create_data.theta)
            Y[i] = int(np.random.random() <= utils.sigmoid(np.dot(dat[i,:], beta)))
            create_data.theta += np.pi/divide_factor
        return dat[:,:d], Y

    create_data.v1 = 2*np.random.randint(0,2,size=d_all)-1
    create_data.v2 = orth_vec(create_data.v1)
    create_data.theta = 0

    x_train, y_train = create_data(n_train)
    x_test, y_test = create_data(n_test)
    
    return x_train, y_train, x_test, y_test    
    
##### Main function to call for running experiment 
def run_exp(dataset="churn", seed=0, design=0, tune=False):
    np.random.seed(seed)
    
    if(dataset == "churn"):
        df = pd.read_csv("data/Churn_Modelling.csv")
        df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
        columns = ['Gender', 'Geography']
        for col in columns:
            one_hot = pd.get_dummies(df[col], drop_first=True)
            df = df.drop(columns=[col])
            df = df.join(one_hot)
        df = df.sample(frac=1) # this shuffles the data
        
        if(design==0):
            x_train, y_train, x_test, y_test = churn_iid(df)
        elif(design==1):
            x_train, y_train, x_test, y_test = churn_order_data_by_location(df)
        elif(design==2):
            x_train, y_train, x_test, y_test = churn_order_data_by_age(df)
        elif(design==3):
            x_train, y_train, x_test, y_test = churn_order_data_by_tenure(df)
        else: 
            assert(False), "Design parameter set incorrectly"
        
        n_val = 1000
        spacing = 500
        twice_spacing = 2*spacing

        
    elif(dataset == "credit"):
        df = pd.read_csv("data/credit.csv")
        df = df.drop(columns=['ID'])
        
        df = df.sample(frac=1) # this shuffles the data
        
        if(design==0):
            x_train, y_train, x_test, y_test = credit_iid(df)
        elif(design==1):
            x_train, y_train, x_test, y_test = credit_order_by_sex(df)
        elif(design==2):
            x_train, y_train, x_test, y_test = credit_order_by_age(df)
        else: 
            assert(False), "Design parameter set incorrectly"

        n_val = 1000
        spacing = 500
        twice_spacing = 2*spacing
         
    elif(dataset == "bank"):
        df = pd.read_csv("data/bank.csv")
        df.loc[df['y']=='yes', 'y'] = 1
        df.loc[df['y']=='no', 'y'] = 0
        df['y'] = df['y'].astype('int')

        df = df.drop(columns=['ID'])
        columns = df.dtypes[df.dtypes != 'int64'][df.dtypes != 'float64'].to_dict().keys()
        for col in columns:
            one_hot = pd.get_dummies(df[col], drop_first=True)
            df = df.drop(columns=[col])
            df = df.join(one_hot, lsuffix=col)

        df = df.sample(frac=1)
        df = df.iloc[:12000]
        if(design==0):
            x_train, y_train, x_test, y_test = bank_iid(df)
        elif(design==1):
            x_train, y_train, x_test, y_test = bank_order_by_age(df)
        elif(design==2):
            x_train, y_train, x_test, y_test = bank_order_by_balance(df)
        else: 
            assert(False), "Design parameter set incorrectly"
            
        n_val = 1000
        spacing = 500
        twice_spacing = 2*spacing
    
    elif(dataset == "fetal"):
        df = pd.read_csv("data/fetal_health.csv")
        df.loc[df['fetal_health']==1, 'fetal_health'] = 0
        df.loc[df['fetal_health']>1, 'fetal_health'] = 1
        df['fetal_health'] = df['fetal_health'].astype('int')

        df = df.sample(frac=1)
        if(design==0):
            x_train, y_train, x_test, y_test = fetal_iid(df)
        elif(design==1):
            x_train, y_train, x_test, y_test = fetal_order_by_acceleration(df)
        else: 
            assert(False), "Design parameter set incorrectly"
            
        n_val = 300
        spacing = 100
        twice_spacing = 2*spacing

    
    elif(dataset == "covariate_shift"):
        n_train = 1000
        n_test = 5000
        x_train, y_train, x_test, y_test = covariate_shift(design, n_train, n_test)
        
        n_val = 1000
        spacing = 500
        twice_spacing = 2*spacing
    
    elif(dataset == "label_shift"):
        n_train = 1000
        n_test = 5000
        x_train, y_train, x_test, y_test = label_shift(design, n_train, n_test)
        
        n_val = 1000
        spacing = 500
        twice_spacing = 2*spacing
        
    elif(dataset == "regression_shift"):
        n_train = 1000
        n_test = 5000
        x_train, y_train, x_test, y_test = regression_shift(design, n_train, n_test)
        
        n_val = 1000
        spacing = 500
        twice_spacing = 2*spacing
    
    else:
        assert(False), "Dataset not supported"

    ############################################
    ### Base model 
    ############################################
    rf = RandomForestClassifier(n_estimators=1000).fit(x_train, y_train)
    rf_preds = rf.predict_proba(x_test)[:,1]
    
    ############################################
    ### Online Platt Scaling (OPS) 
    ############################################
    pred_probs_platt, _, _ = calibration.online_platt_scaling_newton(utils.logit(rf_preds), y_test)
    
    ############################################
    ### Online beta Scaling (OBS) 
    ############################################
    pred_probs_beta, _, _, _ = calibration.online_beta_scaling_newton(utils.log_thresholded(rf_preds), utils.log_thresholded(1-rf_preds), y_test)

    ############################################
    ### OBS + tracking 
    ############################################    
    n_bins = 10
    bin_means = np.linspace(1.0/n_bins, 1.0, n_bins) - 0.05/n_bins
    bin_num = np.ones((n_bins))
    pred_probs_beta_digitized = utils.digitize_predictions_index(pred_probs_beta, n_bins)
    pred_probs_beta_beat = np.zeros((pred_probs_beta.size))
    for i in range(pred_probs_beta_beat.size):
        bin_i = pred_probs_beta_digitized[i]
        pred_probs_beta_beat[i] = bin_means[bin_i]
        bin_means[bin_i] = ((bin_num[bin_i]*bin_means[bin_i]) + y_test[i]
                            )/(bin_num[bin_i]+1)
        bin_num[bin_i] += 1

    ############################################
    ### OBS + hedging 
    ############################################    
    pred_probs_beta_calibeat = np.zeros((pred_probs_beta.size))
    for b in range(n_bins):
        n_bins_calibeat = 10
        b_indices = np.argwhere(pred_probs_beta_digitized==b).squeeze()
        pred_probs_beta_calibeat[b_indices] = utils.randomized_game(
            y_test[b_indices], n_bins_calibeat, b)

    ############################################
    ### Windowed Platt scaling 
    ############################################
    N_window_indices = np.arange(n_val+twice_spacing, rf_preds.size+twice_spacing, spacing).astype('int')
    pred_probs_platt_windowed = np.zeros((rf_preds.size-n_val))
    for n_calib_points in N_window_indices:
        if(True or design==0):
            _, windowed_platt_a, windowed_platt_b = calibration.fit_platt_scaling_parameters(
                utils.logit(rf_preds[:(n_calib_points-twice_spacing)]), 
                y_test[:(n_calib_points-twice_spacing)])            
        else: 
            _, windowed_platt_a, windowed_platt_b = calibration.fit_platt_scaling_parameters(
                utils.logit(rf_preds[(n_calib_points-(n_val+twice_spacing)):(n_calib_points-n_val)]), 
                y_test[(n_calib_points-(n_val+twice_spacing)):(n_calib_points-n_val)])
        
        pred_probs_platt_windowed[(n_calib_points-(n_val+twice_spacing)):(n_calib_points-(n_val+spacing))] = utils.sigmoid(
            windowed_platt_a*utils.logit(rf_preds[(n_calib_points-twice_spacing):(n_calib_points-spacing)]) + windowed_platt_b)
    
    ############################################
    ### Windowed beta scaling 
    ############################################
    N_window_indices = np.arange(n_val+twice_spacing, rf_preds.size+twice_spacing, spacing).astype('int')
    pred_probs_beta_windowed = np.zeros((rf_preds.size-n_val))
    for n_calib_points in N_window_indices:
        if(True or design==0):
            _, windowed_beta_a, windowed_beta_b, windowed_beta_c = calibration.fit_beta_scaling_parameters(
                utils.log_thresholded(rf_preds[:(n_calib_points-twice_spacing)]),
                utils.log_thresholded(1-rf_preds[:(n_calib_points-twice_spacing)]),
                y_test[:(n_calib_points-twice_spacing)])    
        else:
            _, windowed_beta_a, windowed_beta_b, windowed_beta_c = calibration.fit_beta_scaling_parameters(
                utils.log_thresholded(rf_preds[(n_calib_points-(n_val+twice_spacing)):(n_calib_points-n_val)]),
                utils.log_thresholded(1-rf_preds[(n_calib_points-(n_val+twice_spacing)):(n_calib_points-n_val)]),
                y_test[(n_calib_points-(n_val+twice_spacing)):(n_calib_points-n_val)])
        
        pred_probs_beta_windowed[(n_calib_points-(n_val+twice_spacing)):(n_calib_points-(n_val+spacing))] = utils.sigmoid(
            windowed_beta_a*utils.log_thresholded(
                rf_preds[(n_calib_points-twice_spacing):(n_calib_points-spacing)]) + 
            windowed_beta_b*utils.log_thresholded(
                1 - rf_preds[(n_calib_points-twice_spacing):(n_calib_points-spacing)]) + 
            windowed_beta_c)
        
    ############################################
    ### Compute ECE and sharpness
    ############################################
    
    ECE_recalib_platt_windowed = []
    ECE_recalib_beta_windowed = []
    ECE_recalib_platt_continuous = []
    ECE_recalib_beta_continuous = []
    ECE_recalib_beta_continuous_beat = []
    ECE_recalib_beta_continuous_calibeat = []

    SHARPNESS_recalib_platt_windowed = []
    SHARPNESS_recalib_beta_windowed = []
    SHARPNESS_recalib_platt_continuous = []
    SHARPNESS_recalib_beta_continuous = []
    SHARPNESS_recalib_beta_continuous_beat = []
    SHARPNESS_recalib_beta_continuous_calibeat = []
    
    N_calib_points = np.arange(n_val+twice_spacing, rf_preds.size + spacing, spacing).astype('int')
    for n_calib_points in N_calib_points:
        ############################################
        ### Windowed Platt scaling
        ############################################
        test_and_update(ECE_recalib_platt_windowed, assessment.ece, y_test[n_val:n_calib_points], pred_probs_platt_windowed[:(n_calib_points-n_val)])
        test_and_update(SHARPNESS_recalib_platt_windowed, assessment.sharpness, y_test[n_val:n_calib_points], pred_probs_platt_windowed[:(n_calib_points-n_val)])
        
        ############################################
        ### Windowed beta scaling
        ############################################
        test_and_update(ECE_recalib_beta_windowed, assessment.ece, y_test[n_val:n_calib_points], pred_probs_beta_windowed[:(n_calib_points-n_val)])
        test_and_update(SHARPNESS_recalib_beta_windowed, assessment.sharpness, y_test[n_val:n_calib_points], pred_probs_beta_windowed[:(n_calib_points-n_val)])
        
        ############################################
        ### Online Platt Scaling (OPS)
        ############################################
        test_and_update(ECE_recalib_platt_continuous, assessment.ece, y_test[n_val:n_calib_points], pred_probs_platt[n_val:n_calib_points])
        test_and_update(SHARPNESS_recalib_platt_continuous, assessment.sharpness, y_test[n_val:n_calib_points], pred_probs_platt[n_val:n_calib_points])
    
        ############################################
        ### Online beta Scaling (OBS)
        ############################################
        test_and_update(ECE_recalib_beta_continuous, assessment.ece, y_test[n_val:n_calib_points], pred_probs_beta[n_val:n_calib_points])
        test_and_update(SHARPNESS_recalib_beta_continuous, assessment.sharpness, y_test[n_val:n_calib_points], pred_probs_beta[n_val:n_calib_points])
    
        ############################################
        ### Online beta Scaling (OBS) + tracking
        ############################################    
        test_and_update(ECE_recalib_beta_continuous_beat, assessment.ece, y_test[n_val:n_calib_points], pred_probs_beta_beat[n_val:n_calib_points])
        test_and_update(SHARPNESS_recalib_beta_continuous_beat, assessment.sharpness, y_test[n_val:n_calib_points], pred_probs_beta_beat[n_val:n_calib_points])
    
        ############################################
        ### Online beta Scaling (OBS) + calibeating
        ############################################    
        test_and_update(ECE_recalib_beta_continuous_calibeat, assessment.ece, y_test[n_val:n_calib_points], pred_probs_beta_calibeat[n_val:n_calib_points])
        test_and_update(SHARPNESS_recalib_beta_continuous_calibeat, assessment.sharpness, y_test[n_val:n_calib_points], pred_probs_beta_calibeat[n_val:n_calib_points])

    return ECE_recalib_beta_windowed, \
        ECE_recalib_beta_continuous,\
        ECE_recalib_platt_windowed, \
        ECE_recalib_platt_continuous,\
        ECE_recalib_beta_continuous_beat,\
        ECE_recalib_beta_continuous_calibeat,\
        SHARPNESS_recalib_beta_windowed, \
        SHARPNESS_recalib_beta_continuous,\
        SHARPNESS_recalib_platt_windowed, \
        SHARPNESS_recalib_platt_continuous,\
        SHARPNESS_recalib_beta_continuous_beat,\
        SHARPNESS_recalib_beta_continuous_calibeat,\
        N_calib_points
