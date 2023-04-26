# Instructions for use

This repository contains IPython notebooks to reproduce all results from the paper "Online Platt Scaling with Calibeating". The following table identifies each figure from the paper with the corresponding notebook to execute for reproducing that figure. 

| Figure  | Relevant files |
| ------------- | ------------- |
| Figure 2  |  `sec_1_covariate_drift_1d.ipynb` |
| Figure 3, 4, 10, 11  | `sec_1_label_drift_1d.ipynb` , `sec_1_regression_drift_1d.ipynb` |
| Figure 5, 6, 7, 8, 9  | `sec_4_experiments_run.ipynb`, then `sec_4_experiments_plotting.ipynb` |
| Figure 12, 13  | `eps_run.ipynb`, then `eps_plotting.ipynb` |
| Figure 14  | `beta_scaling_experiments_run.ipynb`, then `beta_scaling_plotting.ipynb` |
| Figure 15  | `app_a3_hb_experiments_run.ipynb`, then `app_a3_hb_experiments_plotting.ipynb` |

The notebooks in the latter 4 rows used the `multiprocessing` python library in order to parallelize different runs over 100 cores. 
To reproduce these experiments using fewer cores, go to the corresponding notebook ending in `_run.ipynb` above, and change the parameter `n_core` in the second cell to the number of cores available. 

The datasets have been packaged with the repository and need not be downloaded. On running these notebooks, results and figures will be stored in the folders with prefix `results_`. Results will be stored in subfolders with suffix `_data`, and figures will be stored in subfolders with suffix `_figures`. 

### Standalone implementation of online Platt/beta scaling with calibeating

Standalone implementations of the methods developed in the paper -- OPS, OBS, and their tracking/hedging versions -- are not (yet) directly interfaced. Code for OPS/OBS can be extraced from the notebook `calibration.py`. Code for tracking/hedging can be extracted from `sec_4_experiments_core.ipynb`. If you are interested in contributing standalone versions, please contact me. 

### Acknowledgement

If you use any code from this repository, please acknowledge it. In academic papers, please cite "Online Platt Scaling with Calibeating". 
