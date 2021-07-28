The functions `toplabel_reliability_diagram` and `toplabel_ece` in `assessment.py` implement top-label reliability diagrams and top-lable ECE respectively (these are different from confidence reliability diagrams or confidence ECE proposed by Guo et al. [2]; see the paper [1] for details). 

The function `toplabel_ece` has the following required parameters: 
- `y`: an `N` length vector (1D `numpy` array) with values in `{1, 2, ..., L}`, storing the true labels for `N` points
- `pred_prob`: an `N X L` matrix (2D `numpy` array) of floating point numbers, storing the predicted scores for each of the `N` points for each of the `L` classes

Optionally, `pred_prob` can be an `N` length vector (1D `numpy` array) of floating point numbers, storing the predicted scores for only the predicted class (top-labels). In this case, an additional third parameter `pred_class` must be passed, which is an `N` length vector (1D `numpy` array) with values in `{1, 2, ..., L}`, storing the predicted class labels. The number of bins to be used for ECE estimation can be passed with the parameter `n_bins` (default is 15). 

*Note: Fixed-width binning is used for plotting reliability diagrams. Adaptive binning is used for ECE estimation. However, if the classifier is sufficiently discrete, binning is not used for ECE estimation and the `n_bins` parameter is redundant.*

The function `toplabel_ece` takes the same parameters as above, along with some additional plotting parameters: 
- `ax`: a `matplotlib` axis object, for example the output of `matplotlib.pyplot.subplots()`
- `color` (optional): a [`matplotlib` color](https://matplotlib.org/3.1.1/tutorials/colors/colors.html)


### References

[1] [Top-label calibration](https://arxiv.org/abs/2107.08353)

[2] [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

