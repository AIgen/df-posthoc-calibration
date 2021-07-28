The functions `ece`, `sharpness`, `reliability_diagram`, and `validity_plot` are implemented in `assessment.py`. Each of the following 3 parameters:
- `y`: an `N` length vector (1D `numpy` array) with values in `{0,1}`, storing the true labels for `N` points
- `pred_prob`: an `N` length vector (1D `numpy` array) of floating point numbers, storing the predicted `P(Y=1)` values for each of the `N` points
- `n_bins` (optional): number of bins to be used (default is 10 for `reliability_diagram` and 15 for the other functions)
- `quiet` (optional): print diagnostic output if set to `True` (default: `False`)

The plotting functions `reliability_diagram`, and `validity_plot` take additional plotting parameters: 
- `ax`: a `matplotlib` axis object, for example the output of `matplotlib.pyplot.subplots()`
- `color` (optional): a [`matplotlib` color](https://matplotlib.org/3.1.1/tutorials/colors/colors.html)

All methods use adaptive binning for assessing calibraiton. This behavior can be changed for `reliability_diagram` using the optional `fixed` parameter, which can be set to `True` to use fixed-width bins.
