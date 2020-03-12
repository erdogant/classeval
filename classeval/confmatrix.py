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
    # y_true = y_true.astype(str)
    # y_pred = y_pred.astype(str)
    # Compute confusion matrix
    classnames = np.unique(np.append(y_true, y_pred))
    confmat = confusion_matrix(y_true, y_pred, labels=classnames)

    if normalize:
        if verbose>=3: print("Normalize confusion matrix")
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    if verbose>=3: print(confmat)

    out = {}
    out['classnames'] = classnames
    out['confmat'] = confmat
    out['type'] = 'twoclass'
    out['normalized'] = normalize
    return(out)


# %% Two class evaluation
def twoclass(y_true, y_pred, normalize=False, verbose=3):
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
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    classnames = np.unique(np.append(y_true, y_pred))
    # Compute confusion matrix
    confmat = confusion_matrix(y_true, y_pred, labels=classnames)

    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
        if verbose>=3: print("Normalize confusion matrix")

    if verbose>=3:
        print(confmat)

    out = {}
    out['classnames'] = classnames
    out['confmat'] = confmat
    out['type'] = 'twoclass'
    out['normalized'] = normalize
    return(out)


# %% Multiclass classifier evaluation
def multiclass(y_true, y_pred, normalize=False, verbose=3):
    """Evaluate the results in a multiclass-class model.

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
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    # Compute confusion matrix
    classnames = np.unique(np.append(y_true, y_pred))
    confmat = confusion_matrix(y_true, y_pred, labels=classnames)

    if normalize:
        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    if verbose>=3: print(confmat)

    out = {}
    out['classnames'] = classnames
    out['normalized'] = normalize
    out['confmat'] = confmat
    out['type'] = 'multiclass'
    return(out)


# %% Make the plot
def plot(out, classnames=None, title='', cmap=plt.cm.Blues, figsize=(8,8)):
    """Plot the confusion matrix for the two-class or multi-class model.

    Parameters
    ----------
    out : dict
        Results from twoclass or multiclass function.
    classnames : list of str, optional
        name of the class labels. The default is None.
    title : str, optional
        Title of the figure. The default is ''.
    cmap : object, optional
        colormap. The default is plt.cm.Blues.

    Returns
    -------
    tuple containing (fig, ax).

    """
    if out['type']=='twoclass':
        fig, ax = _plot_twoclass(out['confmat'], title, out['normalized'], cmap, figsize, out['classnames'])
    elif out['type']=='multiclass':
        fig, ax = _plot_multiclass(out['confmat'], title, out['normalized'], cmap, figsize, out['classnames'])
    else:
        fig,ax = None, None

    return(fig, ax)


# %% Multi-class
def _plot_multiclass(confmat, title, normalize, cmap, figsize, classnames):
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        [fig, ax] = plt.subplots()
        im = ax.imshow(confmat, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(confmat.shape[1]),
               yticks=np.arange(confmat.shape[0]),
               xticklabels=classnames,
               yticklabels=classnames,  # label them with the respective list entries
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = confmat.max() / 2.
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(j, i, format(confmat[i, j], fmt),
                        ha="center", va="center",
                        color="white" if confmat[i, j] > thresh else "black")
        fig.tight_layout()

    return(fig, ax)


# %% Two-class
def _plot_twoclass(confmat, title, normalize, cmap, figsize, classnames):
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(confmat, interpolation='nearest', cmap=cmap)
    plt.title(title + 'Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=45)
    plt.yticks(tick_marks, classnames)

    fmt = '.2f' if normalize else 'd'
    thresh = confmat.max() / 2.
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        plt.text(j, i, format(confmat[i, j], fmt), horizontalalignment="center", color="white" if confmat[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.grid(False)

    return(fig, ax)
