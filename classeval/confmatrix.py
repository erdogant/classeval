"""Confusion matrix creating."""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# %%
def _normalize_confmat(confmat, verbose=3):
    if verbose>=4: print("[classeval] >Normalize confusion matrix")
    return confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

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
    # Class names
    class_names = np.unique(np.append(y_true, y_pred))
    # Compute confusion matrix
    confmat = confusion_matrix(y_true, y_pred, labels=class_names)
    # Normalize confmat
    confmat_norm = _normalize_confmat(confmat, verbose=3)
    # Print to screen
    if verbose>=4:
        print(confmat)
    # Output dict
    out = {}
    out['class_names'] = class_names
    out['confmat'] = confmat
    out['confmat_norm'] = confmat_norm
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
    if out['normalized']:
        confmat = out['confmat_norm']
    else:
        confmat = out['confmat']
        
    if len(out['class_names'])==2:
        fig, ax = _plot_twoclass(confmat, title, out['normalized'], cmap, figsize, out['class_names'], fontsize)
    elif len(out['class_names'])>2:
        fig, ax = _plot_multiclass(confmat, title, out['normalized'], cmap, figsize, out['class_names'], fontsize)
    else:
        fig,ax = None, None

    return(fig, ax)


# %% Multi-class
def _plot_multiclass(confmat, title, normalize, cmap, figsize, class_names, fontsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    # Use a perceptually-uniform colormap; fall back to Blues if caller passed one explicitly
    _cmap = plt.cm.Blues if cmap == plt.cm.Blues else cmap
    im = ax.imshow(confmat, interpolation='nearest', cmap=_cmap, aspect='auto')

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    cbar.set_label('Normalised rate' if normalize else 'Count', fontsize=fontsize - 1)

    n = len(class_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax.set_xlabel('Predicted label', fontsize=fontsize, labelpad=10)
    ax.set_ylabel('True label', fontsize=fontsize, labelpad=10)
    _title = (title + ' — ' if title else '') + ('Confusion Matrix (Normalised)' if normalize else 'Confusion Matrix')
    ax.set_title(_title, fontsize=fontsize + 1, fontweight='bold', pad=14)

    fmt = '.2f' if normalize else 'g'
    thresh = confmat.max() / 2.0
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            val = confmat[i, j]
            txt = format(val, fmt)
            # Show percentage below the main number for normalised matrices
            if normalize:
                txt = f'{val:.2f}\n({val*100:.1f}%)'
            ax.text(j, i, txt,
                    ha='center', va='center',
                    color='white' if val > thresh else '#212529',
                    fontsize=fontsize - 1, fontweight='bold' if i == j else 'normal')

    # Highlight diagonal with a subtle border
    for k in range(n):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                              fill=False, edgecolor='#FF6B35', linewidth=2.0)
        ax.add_patch(rect)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


# %% Two-class
def _plot_twoclass(confmat, title, normalize, cmap, figsize, class_names, fontsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    _cmap = plt.cm.Blues if cmap == plt.cm.Blues else cmap
    im = ax.imshow(confmat, interpolation='nearest', cmap=_cmap, aspect='auto')

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    cbar.set_label('Normalised rate' if normalize else 'Count', fontsize=fontsize - 1)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax.set_ylabel('True label', fontsize=fontsize, labelpad=10)
    ax.set_xlabel('Predicted label', fontsize=fontsize, labelpad=10)
    _cm_type = 'Normalised' if normalize else 'Counts'
    _title = (title + ' — ' if title else '') + f'Confusion Matrix ({_cm_type})'
    ax.set_title(_title, fontsize=fontsize + 1, fontweight='bold', pad=14)

    # TP / TN / FP / FN corner labels (only meaningful for 2×2)
    _quadrant_labels = {(0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN', (1, 1): 'TP'}

    fmt = '.2f' if normalize else 'g'
    thresh = confmat.max() / 2.0
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        val = confmat[i, j]
        cell_color = 'white' if val > thresh else '#212529'
        main_txt = format(val, fmt)
        if normalize:
            main_txt = f'{val:.2f}\n({val*100:.1f}%)'
        ax.text(j, i, main_txt,
                ha='center', va='center',
                color=cell_color, fontsize=fontsize,
                fontweight='bold' if i == j else 'normal')
        # Corner tag
        if (i, j) in _quadrant_labels:
            ax.text(j + 0.42, i - 0.42, _quadrant_labels[(i, j)],
                    ha='right', va='top',
                    color=cell_color, fontsize=fontsize - 3, style='italic')

    # Diagonal highlight
    for k in range(len(class_names)):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                              fill=False, edgecolor='#FF6B35', linewidth=2.5)
        ax.add_patch(rect)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig, ax
