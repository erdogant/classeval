"""ROC is for computing receiver operator characteristics for two-class and multi-class models."""
# -----------------------------------------------------
# Name        : classeval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/classeval
# Licence     : MIT
# -----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import colourmap


# %% Evaluate ROC
def eval(y_true, y_proba, y_score=None, pos_label=None, threshold=0.5, verbose=3):
    """Receiver operator Curve.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    y_score : array of floats
        decision_function for the predicted labels. (only required in case of multi-class)
    pos_label : str
        Positive label (only for the two-class models and when y_ture is of type string)
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    """
    if len(np.unique(y_true))==2:
        out = _ROC_twoclass(y_true, y_proba, threshold=threshold, pos_label=pos_label, verbose=verbose)
    elif len(np.unique(y_true))>2:
        out = _ROC_multiclass(y_true, y_proba, y_score, threshold=threshold, verbose=verbose)
        out['roc_auc '] = AUC_multiclass(y_true, y_proba, verbose=verbose)
    else:
        return None

    return(out)


# %% ROC plot
def plot(out, ax=None, title='', fontsize=12, figsize=(12,8), verbose=3):
    """Plot ROC curves.

    Parameters
    ----------
    out : dict
        results from the eval() function.
    title : str, optional
        Title of the figure. The default is ''.
    ax : figure object, optional
        Figure axis. The default is None.
    fontsize : int, optional
        Size of the fonts. The default is 12.
    figsize : tuple, optional
        Figure size. The default is (12,8).
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    tuple containing (fig,ax).

    """
    if len(out['class_names'])==2:
        ax = _plot_twoclass(out, fontsize=fontsize, title=title, ax=ax, figsize=figsize)
    elif len(out['class_names'])>2:
        ax = _plot_multiclass(out, fontsize=fontsize, title=title, ax=ax, figsize=figsize, cmap='Set1')
    else:
        ax = None

    return(ax)


# %% ROC plot for multi-class model
def _plot_multiclass(out, ax=None, fontsize=12, title='', figsize=(12,8), cmap='Set1'):
    fpr = out['fpr']
    tpr = out['tpr']
    roc_auc = out['auc']
    colors = colourmap.generate(len(out['class_names']), cmap=cmap)
    linewidth = 1.5

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)

    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=2)

    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=2)

    for i, [class_name, color] in enumerate(zip(out['class_names'], colors)):
        ax.plot(fpr[i], tpr[i], color=color, lw=linewidth, label=('ROC curve of class %s (area = %.2f)' '' %(class_name, roc_auc[i])))

    ax.plot([0, 1], [0, 1], 'k--', lw=linewidth)
    ax.set_xlabel('False Positive Rate', fontsize=fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=fontsize)
    ax.set_title('Extension of Receiver operating characteristic to multi-class', fontsize=fontsize)
    ax.legend(loc="lower right", fontsize=fontsize)
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    # plt.show()
    # Return
    return ax


# %% ROC plot for two-class model
def _plot_twoclass(out, fontsize=12, title='', ax=None, figsize=(12,8)):
    """Plot two-class ROC curve.

    Parameters
    ----------
    fpr : array-like
        False positive rate.
    tpr : array-like
        True positive rate.
    roc_auc : float
        AUC scoring.
    fontsize : int, optional
        Size of the fonts. The default is 12.
    title : str, optional
        Title of the figure. The default is ''.
    ax : figure object, optional
        Figure axis. The default is None.
    figsize : tuple, optional
        Figure size. The default is (12,8).

    Returns
    -------
    ax : object
        Axis.

    """
    fpr = out['fpr']
    tpr = out['tpr']
    roc_auc = out.get('auc',None)

    if ax is None:
        fig,ax= plt.subplots(figsize=figsize)

    linewidth = 1.5
    ax.plot(fpr, tpr, color='darkorange', lw=linewidth, label='ROC curve (area = %.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=fontsize)
    ax.set_title('[%s] Receiver operating characteristic. AUC:%.3f' %(title, roc_auc), fontsize=fontsize)
    ax.legend(loc="lower right", fontsize=fontsize)
    ax.grid(True)
    return ax


# %% ROC plot
def _ROC_multiclass(y_true, y_proba, y_score, threshold=0.5, title='', ax=None, figsize=(12,8), fontsize=12, verbose=3):
    if y_score is None:
        print('[classeval] ROC multiclass should have y_score! <y_score = model.decision_function(X)>')
        return None
    if not len(y_true)==len(y_proba):
        raise Exception('[classeval] ROC multiclass should have same number of y_true and y_proba rows!')
    if not len(y_true)==y_score.shape[0]:
        raise Exception('[classeval] ROC multiclass should have same number of y_true and y_score rows!')

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    class_names=np.unique(y_true)
    y_true = label_binarize(y_true, classes=class_names)
    n_classes = y_true.shape[1]

    # Compute AUC per class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    out = {}
    out['fpr'] = fpr
    out['tpr'] = tpr
    out['auc'] = roc_auc
    out['roc_auc_macro'] = roc_auc["macro"]
    out['roc_auc_micro'] = roc_auc["micro"]
    out['class_names'] = class_names
    return(out)


# %% AUC multi-class
def AUC_multiclass(y_true, y_proba, verbose=3):
    """AUC scoring for multiclass predictions.

    Description
    -----------
    Calculate the AUC using the One-vs-Rest scheme (OvR) and One-vs-One scheme (OvO) schemes.
    The multi-class One-vs-One scheme compares every unique pairwise combination of classes.
    Macro average, and a prevalence-weighted average.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    """
    macro_roc_auc_ovo = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_true, y_proba, multi_class="ovo", average="weighted")
    macro_roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    if verbose>=3:
        print("[classeval] One-vs-One ROC AUC scores:\n   {:.6f} (macro),\n   {:.6f} " "(weighted by prevalence)" .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
        print("[classeval] One-vs-Rest ROC AUC scores:\n   {:.6f} (macro),\n   {:.6f} " "(weighted by prevalence)" .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    out = {}
    out['macro_roc_auc_ovo'] = macro_roc_auc_ovo
    out['weighted_roc_auc_ovo'] = weighted_roc_auc_ovo
    out['macro_roc_auc_ovr'] = macro_roc_auc_ovr
    out['weighted_roc_auc_ovr'] = weighted_roc_auc_ovr
    return(out)


# %% ROC plot
def _ROC_twoclass(y_true, y_proba, pos_label=None, threshold=0.5, verbose=3):
    """Receiver operator Curve.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.
    title : str, optional
        Title of the figure. The default is ''.
    ax : figure object, optional
        Figure axis. The default is None.
    showfig : bool, optional
        Show the figure. The default is True.
    fontsize : int, optional
        Fontsize in the figures. The default is 12.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    """
    # ROC
    [fpr, tpr, thresholds] = roc_curve(y_true, y_proba, pos_label=pos_label)
    # AUC
    roc_auc = auc(fpr, tpr)

    out = {}
    out['auc'] = roc_auc
    out['thresholds'] = thresholds
    out['fpr'] = fpr
    out['tpr'] = tpr
    out['class_names'] = np.unique(y_true)

    return(out)
