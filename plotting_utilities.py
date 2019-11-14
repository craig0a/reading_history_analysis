import base64
import io
import jinja2
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.metrics import auc, roc_curve

def plot_roc_curve(true_response, pred_responses, labels, ax, fnr_threshold = 0.2):
    """
    Create a curve showing false positive vs. false negative rates for different classification thresholds

    Arguments:
        true_response : array, shape = [n_samples]
            True binary labels. If labels are not either {-1, 1} or {0, 1}, then
            pos_label should be explicitly given.
        pred_responses : array, shape = [n_samples, number of predictors]
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by "decision_function" on some classifiers).
        labels : list of strings, shape = [number of predictors]
            The labels of the predictors.
        ax : A matplotlib.axes.Axes instance

    Returns:
        thresholds : threshold for which false negative rate first below specified value
    """
    for pred in range(0, np.shape(pred_responses)[1]):
        fpr, tpr, thresholds = roc_curve(true_response, pred_responses[:, pred])
        fnr = 1-tpr
        fpr_fnr_auc = auc(fpr, fnr)
        ax.plot(fpr, fnr,label='%s: %0.2f' %(labels[pred], fpr_fnr_auc))

        i = next(x[0] for x in enumerate(fnr) if x[1] < fnr_threshold)
        ax.text(fpr[int(i)], fnr[int(i)], '%.2f'%(thresholds[int(i)]))
        ax.plot(fpr[int(i)], fnr[int(i)], 'ko')

    ax.set_title('Adjusted ROC curve')
    ax.set_ylabel('False negative rate')
    ax.set_xlabel('False positive rate')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    return thresholds[int(i)]




#https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels,
            row_title = '', col_title = '', ax=None,
            cbar_kw={}, cbarlabel="", vmin=None, vmax=None,
            x_tick_rotation = 0, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    Returns:
        im, cbar : handles to image and colorbar

    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()
        
    if not vmin:
        vmin = min(data)
        
    if not vmax:
        vmax = max(data)

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=x_tick_rotation, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=True, left=True)

    ax.set_xlabel(col_title)
    ax.set_ylabel(row_title)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.
    Returns:
        texts: list of text annotations

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def model_hyperparameter_tuning_plot(balanced_accuracy, passed_fail, failed_pass,
                                     model_size, N_train, x, x_label = ''):
    """
    Set of line plots showing balanced_accuracy, false positive rate (passed_fail), false negative rate (failed_pass) as a percentage of the number of samples (N_train), and model memory size for different values of the hyperparameters used in a model (x)

    Arguments:
        balanced_accuracy : list of float, [n_models]
        passed_fail : list of float, [n_models]
        failed_pass : list of float, [n_models]
        model_size : list of float, [n_models]
        N_train : number of samples used to calculate statistics
        x : list of float, [n_models]
    Returns:
        None
    """
    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharey = False, sharex = True, figsize = (8*ncols,3*nrows))
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=2.0)
    ax = ax.reshape(nrows, ncols)

    ax[0,0].plot(x, balanced_accuracy, color = 'k')
    ax[0,0].set_ylabel('Balanced accuracy')
    ax[0,0].grid()
    ax[0,0].set_xlabel(x_label)

    ax[0,1].plot(x, np.divide(passed_fail,N_train)*100., color = 'r', label = 'False positive (inauthentic labeled authentic)')
    ax[0,1].plot(x, np.divide(failed_pass,N_train)*100, color = 'b', label = 'False negative (authentic labeled inauthentic)')
    ax[0,1].set_ylabel('Misclassifications\n[% of total samples]')
    ax[0,1].legend()
    ax[0,1].grid()
    ax[0,1].set_xlabel(x_label)

    ax[1,0].plot(x, np.divide(model_size,1e6))
    ax[1,0].set_ylabel('Model pickle size [MB]')
    ax[1,0].grid()
    ax[1,0].set_xlabel(x_label)

def plot_grid_search(results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    """
     Plot outputs from sklearn GridSearch over two parameters

     Arguments:
         results : GridSearch results
         grid_param_1 :  array of first search parameter values
         grid_param_2 :  array of second search parameter values
         name_param_1 :  name of first search parameter values
         name_param_2 :  name of second search parameter values
     Returns:
         None
     """

    # Plot Grid search scores
    fig, ax = plt.subplots(1,3, figsize = (16,4))
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=2.0)

    mean_score = np.array(results['mean_test_score']).reshape(len(grid_param_2),len(grid_param_1))
    std_score = np.array(results['std_test_score']).reshape(len(grid_param_2),len(grid_param_1))
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax[0].plot(grid_param_1, mean_score[idx,:], '-o', label= name_param_2 + ': ' + str(val))
        ax[0].fill_between(grid_param_1, mean_score[idx,:]-std_score[idx,:],
                           mean_score[idx,:]+std_score[idx,:],  alpha=0.1)

    ax[0].set_title('Mean score', fontsize=16)
    ax[0].set_xlabel(name_param_1, fontsize=16)
    ax[0].grid('on')

    mean_fit_time = np.array(results['mean_fit_time']).reshape(len(grid_param_2),len(grid_param_1))
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax[1].plot(grid_param_1, mean_fit_time[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax[1].set_title('Mean fit time', fontsize=16)
    ax[1].set_xlabel(name_param_1, fontsize=16)

    ax[1].grid('on')

    mean_score_time = np.array(results['mean_score_time']).reshape(len(grid_param_2),len(grid_param_1))
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax[2].plot(grid_param_1, mean_score_time[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax[2].set_xlabel(name_param_1, fontsize=16)
    ax[2].set_title('Mean score time', fontsize=16)
    ax[2].legend( bbox_to_anchor=(1.05, 0.75), fontsize=15)
    ax[2].grid('on')

def jointplot(x,y, c = 'k', cmap = 'gray',
              xmin = 0, xmax = 1, ymin = 0, ymax = 1, delta = 0.05,
              gridsize = 50,
              joint_xlabel = '', joint_ylabel = '',
              marginal_xlabel = '', marginal_ylabel = ''):
    """
    joint plot of two continuous features; effectively a replacement of seaborn jointplot

    Arguments:
        x,y : arrays of floats, shape = [n_samples]
    Returns:
        fig, ax_joint, ax_marg_x, ax_marg_y : handles to figure and each of the three subplot axes
    """


    fig = plt.figure()
    gs = GridSpec(4,4)

    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])

    ax_joint.hexbin(x,y, cmap = cmap, bins= 'log', gridsize = gridsize )
    ax_joint.axis([xmin-delta, xmax+delta, ymin-delta, ymax+delta])

    ax_marg_x.hist(x, density = False, color = c, range = (xmin, xmax), align = 'mid')
    ax_marg_x.set_xlim([xmin-delta, xmax+delta])
    ax_marg_y.hist(y, density = False, color = c, range = (xmin, xmax), align = 'mid', orientation="horizontal")
    ax_marg_y.set_ylim([ymin-delta, ymax+delta])

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel(joint_xlabel)
    ax_joint.set_ylabel(joint_ylabel)

    # Set labels on marginals
    ax_marg_y.set_xlabel(marginal_xlabel)
    ax_marg_x.set_ylabel(marginal_ylabel )

    return fig, ax_joint, ax_marg_x, ax_marg_y