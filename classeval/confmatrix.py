"""Confusion matrix creating."""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


# %% Class evaluation
def eval(y_true, y_pred, normalize=False, verbose=3):
    """Evaluate the results in a two-class model.

    Parameters
    ----------
    y_true : array-like
        True labels of the classes.
    y_pred : array-like
        Predicted labels.
    normalize : bool, optional
        Normalize the values in the confusion matrix. The default is False.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    """
    # Compute confusion matrix
    class_names = np.unique(np.append(y_true, y_pred))
    confmat = confusion_matrix(y_true, y_pred, labels=class_names)

    if normalize:
        if verbose>=3: print("Normalize confusion matrix")
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    if verbose>=4:
        print(confmat)

    out = {}
    out['class_names'] = class_names
    out['confmat'] = confmat
    out['normalized'] = normalize
    return(out)


# %% Make the plot
def plot(out, class_names=None, title='', cmap=plt.cm.Blues, figsize=(12,12), fontsize=14):
    """Plot the confusion matrix for the two-class or multi-class model.

    Parameters
    ----------
    out : dict
        Results from twoclass or multiclass function.
    class_names : list of str, optional
        name of the class labels. The default is None.
    title : str, optional
        Title of the figure. The default is ''.
    cmap : object, optional
        colormap. The default is plt.cm.Blues.

    Returns
    -------
    tuple containing (fig, ax).

    """
    if len(out['class_names'])==2:
        fig, ax = _plot_twoclass(out['confmat'], title, out['normalized'], cmap, figsize, out['class_names'], fontsize)
    elif len(out['class_names'])>2:
        fig, ax = _plot_multiclass(out['confmat'], title, out['normalized'], cmap, figsize, out['class_names'], fontsize)
    else:
        fig,ax = None, None

    return(fig, ax)


# %% Multi-class
def _plot_multiclass(confmat, title, normalize, cmap, figsize, class_names, fontsize):
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(confmat, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks
    ax.set(xticks=np.arange(confmat.shape[1]),
            yticks=np.arange(confmat.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,  # label them with the respective list entries
            title=title,
            ylabel='True label',
            xlabel='Predicted label',
            )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confmat.max() / 2.
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(j, i, format(confmat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confmat[i, j] > thresh else "black",
                    fontsize=fontsize,
                    )

    ax.grid(False)
    fig.tight_layout()

    return(fig, ax)


# %% Two-class
def _plot_twoclass(confmat, title, normalize, cmap, figsize, class_names, fontsize):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(confmat, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    tick_marks = np.arange(len(class_names))

    ax.set(title=title + 'Confusion matrix',
            xticks=tick_marks,
            yticks=tick_marks,
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label',
            )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize)

    # Numbers in the figure
    fmt = '.2f' if normalize else 'd'
    thresh = confmat.max() / 2.
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        plt.text(j, i, format(confmat[i, j], fmt), 
                 horizontalalignment="center", 
                 color="white" if confmat[i, j] > thresh else "black", 
                 fontsize=fontsize,
                 )

    ax.grid(False)
    fig.tight_layout()

    return(fig, ax)
