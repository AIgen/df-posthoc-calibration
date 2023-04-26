# Distribution-free, model-agnostic, posthoc calibration 
Any probabilistic classification model can be provably posthoc calibrated, for arbitrarily distributed data [[3]](https://arxiv.org/abs/2006.10564). This repository contains an easy-to-use, low-dependency, python library that achieves this goal for multiclass [top-label calibration](#Top-label-calibration) [[1]](https://arxiv.org/abs/2107.08353) and [binary calibration](binary-calibration) [[2]](https://arxiv.org/abs/2105.04656). 

The simplest use case is to recalibrate an existing probabilistic classification model, called the base model. The base model can be trained using any library in any programming language. Our code is agnostic to the details of the model and works on top of the final class probabilities predicted by the model, which can simply be loaded from a file. This is also called the posthoc calibration setting. 

The library was developed on a UNIX system using `Python 3.9.1`, but it should work with other versions of Python3. The only other dependency is the library `numpy`. Running the illustrative examples (`.ipynb` files) requires Jupyter Notebook and two additional libraries, `scikit-learn` and `matplotlib`.

## Top-label calibration
Top-label calibration is a practically useful adaptation of confidence calibration [[7]](https://arxiv.org/abs/1706.04599); see the paper [[1]](https://arxiv.org/abs/2107.08353) for further details. The class `HB_toplabel` in `calibration.py` implements top-label histogram binning. To use this class, first load or compute the following two objects: 
- `base_probs`: an `N X L` matrix (2D `numpy` array) of floating point numbers, storing the predicted scores for each of the `N` calibration points for each of the `L` classes, using an arbitrary base model
- `true_labels`: an `N` length vector (1D `numpy` array) with values in `{1, 2, ..., L}`, storing the true labels for each of the `N` calibration points

A histogram binning wrapper can be learnt around the base model using **3 lines of code**:
```python
from calibration import HB_toplabel
hb = calibration.HB_toplabel(points_per_bin=50)
hb.fit(base_probs, true_labels)
```
That's it, `hb` can now be used to make top-label calibrated predictions. Let the base model score matrix on some new (test) data be `base_probs_test`, a 2D `numpy` vector of floats, of size `N_test X L`. Then
```python
calibrated_probs_test = hb.predict_proba(base_probs_test)
```
gives the calibrated probabilities for the predicted classes (a 1D `numpy` vector of floats of length `N_test`). Note that the corresponding predicted classes are given by:
```python
predicted_class_test = np.argmax(base_probs_test, axis=1) + 1
```
Here the `+ 1` ensures that the final class predictions are in `{1, 2, ..., L}`.

### Self-contained example with ResNet-50 on CIFAR10
The file `example_cifar10.ipynb` documents an illustrative example for achieve top-label calibrated predictions on the CIFAR10 dataset [[5]](https://www.cs.toronto.edu/~kriz/cifar.html). First, a pretrained ResNet-50 model from the `focal_calibration` repository [[6]](https://github.com/torrvision/focal_calibration) was used to produce a base prediction matrix. The logits corresponding to these predictions are stored in `data/cifar10_resnet50/`. Along with these, the logits corresponding to the same model post temperature scaling are also stored. The file `example_cifar10.ipynb` loads these logits, computes the corresponding predicted probabilities, and top-label recalibrates them as illustrated above. The final top-label reliability diagram, and top-label ECE corresponding to the ResNet-50 model, temperature scaling model, and histogram binning model are reproduced below. 

<div style="text-align: center;">
  <img src="figs/cifar10_top_label.png?raw=true" width="700" /> 
</div>

The plots show that histogram binning improves the top-label calibration of the ResNet-50 model more than temperature scaling. Further details and references for these plots can be found in the paper [[1]](https://arxiv.org/abs/2107.08353). The paper also contains extensive experimentation with additional datasets and deepnet architectures. The code used to make these plots and compute the ECE can be found in `assessment.py` ([documentation here](docs/toplabel_assessment.md)).

## Binary calibration
The class `HB_binary` in `calibration.py` implements binary histogram binning. To use this class, first load or compute the following two objects: 
- `base_probs`: an `N` length vector (1D `numpy` array) of floating point numbers, storing the predicted `P(Y=1)` values for each of the `N` calibration points, using an arbitrary base model
- `true_labels`: an `N` length vector (1D `numpy` array) of 0s and 1s, storing the true labels for each of the `N` calibration points

A histogram binning wrapper can be learnt around the base model using **3 lines of code**:
```python
from calibration import HB_binary
hb = HB_binary(n_bins=15)
hb.fit(base_probs, true_labels)
```
That's it, `hb` can now be used to make calibrated predictions. Let the base model probabilities on some new data be `base_probs_test` (a 1D `numpy` vector of floats). Then
```python
calibrated_probs_test = hb.predict_proba(base_probs_test)
```
gives the calibrated probabilities (a 1D `numpy` vector of floats).

### Self-contained example with logistic regression
The file `example_credit.ipynb` documents an illustrative example for learning and recalibrating a logistic regression classifier on the credit default dataset [[4]](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). For the full pipeline, the dataset is first split into three parts (training, calibration, and test). The salient lines of code (paraphrased) are:

```python
x_train, y_train, x_calib, y_calib, x_test, y_test = load_data_and_create_splits()

lr = sklearn.LogisticRegression().fit(x_train, y_train)

lr_calib = lr.predict_proba(x_calib)[:,1]
hb = HB_binary(n_bins=15)
hb.fit(pred_probs_calib, y_calib)

lr_test = lr.predict_proba(x_test)[:,1]
hb_test = hb.predict_proba(lr_test)
```

The `numpy` array `hb_test` contains the calibrated probabilities on the test data. The file `binary_assessment.py` contains four assessment metrics for calibration: reliability diagrams, validity_plots, ECE, and sharpness ([documentation here](docs/binary_assessment.md)). Some plots from `example_credit.ipynb` are reproduced below: 

<div style="text-align: center;">
  <img src="figs/logistic_regression.png?raw=true" width="350" />
  <img src="figs/histogram_binning.png?raw=true" width="350" /> 
</div>
<!---![](logistic_regression.png?raw=true) ![](histogram_binning.png?raw=true)--->

The plots show that histogram binning improves the calibration of logistic regression. Further details and references for these plots can be found in the paper [[2]](https://arxiv.org/abs/2105.04656). 

## License
This repository is licensed under the terms of the [MIT non-commercial License](LICENSE).

## Acknowedgement
If you use any code from this repository, please acknowledge it by referring to the github page and/or citing relevant papers [1, 2, 3]. 

## References

[1] [Top-label calibration](https://arxiv.org/abs/2107.08353)

[2] [Distribution-free calibration guarantees for histogram binning without sample splitting](https://arxiv.org/abs/2105.04656)

[3] [Distribution-free binary classification: prediction sets, confidence intervals and calibration](https://arxiv.org/abs/2006.10564)

[4] [Credit default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

[5] [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

[6] [Focal loss repository](https://github.com/torrvision/focal_calibration)

[7] [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
