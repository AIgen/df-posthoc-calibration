# Distribution-free, model-agnostic, posthoc calibration 
Any probabilistic classification model can be provably posthoc calibrated, even if the data is anomalously distributed [1,2,3]. This repository contains easy-to-use code that achieves this goal for binary classification. Code for multiclass classification will be released shortly.

## Binary calibration
The class `HB_binary` in `calibration.py` contains the binary calibrator (histogram binning). The file `assessment.py` contains some common calibration assessment functions such as reliability diagrams and ECE. 

The file `credit_default_example.ipynb` documents an illustrative example for recalibrating a logistic regression classifier on the credit default dataset [4]. Additional details will be provided here soon. 

[1] [Top-label calibration](https://arxiv.org/abs/2107.08353)

[2] [Distribution-free calibration guarantees for histogram binning without sample splitting](https://arxiv.org/abs/2105.04656)

[3] [Distribution-free binary classification: prediction sets, confidence intervals and calibration](https://arxiv.org/abs/2006.10564)

[4] [Credit default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
