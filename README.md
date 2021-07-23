# Distribution-free, model-agnostic, posthoc calibration 
Any probabilistic classification model can be provably posthoc calibrated, for arbitrarily distributed data [1,2,3]. This repository contains an easy-to-use python library that achieves this goal for binary classification. *Code for multiclass classification will be released shortly.*

The simplest use case is to recalibrate an existing probabilistic classification model, called the base model. The base model can be trained using any library in any programming language. Our code is agnostic to the details of the model and works on top of the final class probabilities predicted by the model, which can simply be loaded from a file. This is also called the posthoc calibration setting. 


## Binary calibration
Let ``base_probs`` be a 1-D numpy array of floats storing the predicted P(Y=1) values from a base model, and ``labels`` be a 1-D numpy array of 0s and 1s storing the true labels (both arrays should have matching length).  A histogram binning wrapper can be learnt around the base model using **3 lines of code**:
```python
from calibration import HB_binary
hb = HB_binary(n_bins=15)
hb.fit(base_probs, true_labels)
```
That's it, histogram binning can now be used to make calibrated predictions. Let the base model probabilities on some new data be ``base_probs_test`` (a 1-D numpy vector of floats). Then
```python
calibrated_probs_test = hb.predict_proba(base_probs_test)
```
gives the calibrated probabilities (a 1-D numpy vector of floats).

### Self-contained example with logistic regression
The file `credit_default_example.ipynb` documents an illustrative example for learning and recalibrating a logistic regression classifier on the credit default dataset [4]. For the full pipeline, the dataset is first split into three parts (training, calibration, and test). The salient lines of code (paraphrased) are:

```python
x_train, y_train, x_calib, y_calib, x_test, y_test = load_data_and_create_splits()

lr = sklearn.LogisticRegression().fit(x_train, y_train)

lr_calib = lr.predict_proba(x_calib)[:,1]
hb = HB_binary(n_bins=15)
hb.fit(pred_probs_calib, y_calib)

lr_test = lr.predict_proba(x_test)[:,1]
hb_test = hb.predict_proba(lr_test)
```
``hb_test`` contains the calibrated probabilities on the test data. The file `binary_assessment.py` contains four assessment metrics for calibration: reliability diagrams, validity_plots, ECE, and sharpness. Some plots from `credit_default_example.ipynb` are reproduced below: 

<p float="left">
  <img src="logistic_regression.png?raw=true" width="250" />
  <img src="histogram_binning.png?raw=true" width="250" /> 
</p>
<!---![](logistic_regression.png?raw=true) ![](histogram_binning.png?raw=true)--->

The plots show that histogram binning improves the calibration of logistic regression. Further details and references for these plots can be found in the paper [1]. 

## License
This repository is licensed under the terms of the [MIT non-commercial License](LICENSE).

## References

[1] [Top-label calibration](https://arxiv.org/abs/2107.08353)

[2] [Distribution-free calibration guarantees for histogram binning without sample splitting](https://arxiv.org/abs/2105.04656)

[3] [Distribution-free binary classification: prediction sets, confidence intervals and calibration](https://arxiv.org/abs/2006.10564)

[4] [Credit default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
