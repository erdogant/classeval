"""classeval is for fast and easy classifier evaluation."""
# -----------------------------------------------------
# Name        : classeval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/classeval
# Licence     : MIT
# -----------------------------------------------------

import colourmap
import classeval.confmatrix as confmatrix
import classeval.ROC as ROC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score
from funcsigs import signature


# %% Plot crossvalidation results for two class models.
def plot_cross(out, title='', fontsize=12, figsize=(15, 8)):
    """ Plot crossvalidation results for two class models.

    Parameters
    ----------
    out : dict
        dictionary containing multiple evaluated models from the eval() function.
    title : str, optional
        Title of the figure. The default is ''.
    fontsize : int, optional
        Font-size. The default is 12.
    figsize : tuple, optional
        Figure size. The default is (20,15).

    Returns
    -------
    fig, ax

    """
    colors = colourmap.generate(len(out.keys()))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#F8F9FA')

    get_auc = []
    all_fpr = np.linspace(0, 1, 200)
    tpr_matrix = []

    for i, key in enumerate(out.keys()):
        fold_out = out.get(key)
        ax = ROC.plot(fold_out, label='', color=(*colors[i, :3], 0.55), ax=ax,
                      title=title, fontsize=fontsize, verbose=0)
        get_auc.append(fold_out['auc'])
        tpr_matrix.append(np.interp(all_fpr, fold_out['fpr'], fold_out['tpr']))

    # Mean ± std band
    tpr_arr = np.array(tpr_matrix)
    mean_tpr = tpr_arr.mean(axis=0)
    std_tpr = tpr_arr.std(axis=0)
    mean_auc = float(np.mean(get_auc))
    std_auc = float(np.std(get_auc))

    ax.fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    alpha=0.18, color='#1A237E', label='± 1 std. dev.')
    ax.plot(all_fpr, mean_tpr, color='#1A237E', lw=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    _cv_title = (title + ' — ' if title else '') + 'Cross-validation ROC'
    ax.set_title(_cv_title, fontsize=fontsize + 1, fontweight='bold', pad=12)

    # Reorder legend so mean appears first
    handles, labels = ax.get_legend_handles_labels()
    mean_idx = [i for i, l in enumerate(labels) if 'Mean ROC' in l]
    std_idx  = [i for i, l in enumerate(labels) if '± 1 std' in l]
    other_idx = [i for i in range(len(labels)) if i not in mean_idx + std_idx]
    order = mean_idx + std_idx + other_idx
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='lower right', fontsize=fontsize - 1, framealpha=0.9)

    return fig, ax

# %% Main function for all two class results.
def plot(out, title='', fontsize=12, figsize=(20, 15), bar_edgecolor='black'):
    """Make plot based on evaluated model.

    Parameters
    ----------
    out : dict
        Evaluated model from the eval() function.
    title : str, optional
        Title of the figure. The default is ''.
    fontsize : int, optional
        Font-size. The default is 12.
    figsize : tuple, optional
        Figure size. The default is (20,15).
    bar_edgecolor : str or None, optional
        Outline colour of the bars in the stacked bar chart. The default is
        ``'black'``. Pass ``None`` for no outline.

    Returns
    -------
    ax : Object

    """
    y_true = out['y_true']
    y_proba = out['y_proba']
    threshold = out.get('threshold', None)

    if len(out['class_names']) == 2:
        # Setup figure
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        # pr curve
        AP(y_true, y_proba, fontsize=fontsize, title=title, ax=ax[1][0], showfig=True)
        # ROC plot
        _ = ROC.plot(out, fontsize=fontsize, title=title, ax=ax[0][0], verbose=0)
        # CAP plot
        _ = CAP(y_true, y_proba, fontsize=fontsize, ax=ax[0][1], showfig=True)
        # Probability plot
        _ = TPFP(y_true, y_proba, fontsize=fontsize, threshold=threshold, title=title, ax=ax[1][1], showfig=True)
        # Show plot
        plt.show()
        # Confusion matrix
        _ = confmatrix.plot(out['confmat'], title=title)
        # Stacked bar data
        class_names = {True: str(out['pos_label']), False: str(out['neg_label'])}
        y_true_str = np.array(list(map(class_names.get, y_true)))
        y_pred_str = np.array(list(map(class_names.get, out['y_pred'])))
        # Make stackedbar plot
        _ = _stackedbar_multiclass(y_true_str, y_pred_str, showfig=True, fontsize=fontsize, bar_edgecolor=bar_edgecolor)
    elif len(out['class_names'])>2:
        ax = None
        _ = ROC.plot(out['ROCAUC'], fontsize=fontsize)
        _ = confmatrix.plot(out['confmat'])
        _ = _stackedbar_multiclass(out['y_true'], out['y_pred'], showfig=True, fontsize=fontsize, bar_edgecolor=bar_edgecolor)

    return fig, ax


# %% Main function for all two class results.
def eval(y_true, y_proba, y_score=None, y_pred=None, pos_label=None, threshold=0.5, normalize=False, verbose=3):
    """Evaluate and make plots for two-class models.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    y_score : array of floats
        decision_function for the predicted labels. (only required in case of multi-class)
    y_pred : array-like
        Predicted labels from model.
    pos_label : str
        Positive label (only for the two-class models and when y_true is of type string. If you set bool, then the positive label is True)
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.
    normalize : bool, optional
        Normalize the values in the confusion matrix. The default is False.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    Output is a dict containing results that are based on ``eval_twoclass`` or ``eval_multiclass``.

    """
    if isinstance(y_proba, pd.DataFrame):
        print('[classeval] pandas DataFrame not allowed as input for y_proba. Use lists or numpy array-like')
    if isinstance(y_true, pd.DataFrame):
        print('[classeval] pandas DataFrame not allowed as input for y_true. Use lists or numpy array-like')

    if (pos_label is None) and (str(y_true.dtype) == 'bool'):
        pos_label = True
    elif pos_label is None:
        pos_label = np.unique(y_true)[0]
        if verbose>=2: print('[classeval] >Warning In two class evaluation, it is recomended to specify <pos_label>. pos_label is set to "%s"' %(pos_label))
    if pos_label is not None:
        if not np.any(np.isin(y_true, pos_label)):
            raise Exception(['[classeval] pos_label is not found in y_true!'])

    # Create classification report
    class_names = np.unique(y_true)
    if len(class_names)==2:
        if len(np.shape(y_proba))>1:
            print('[classeval] Warning >In two class evaluation, it is only recomended to specify <y_proba> of interest. The first column is now taken as input.')
            y_proba = y_proba[:,0]
        out = eval_twoclass(y_true, y_proba, threshold=threshold, pos_label=pos_label, normalize=normalize, verbose=verbose)
    elif len(class_names)>2:
        out = eval_multiclass(y_true, y_proba, y_score, y_pred, normalize=normalize, verbose=verbose)
    else:
        print('[classeval] The input variable [y_true] should contain at least two classes.')
        out=None

    # Return
    return(out)


# %% Two class results
def eval_multiclass(y_true, y_proba, y_score, y_pred, normalize=False, verbose=3):
    """Evaluate for multi-class model.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    y_score : array of floats
        decision_function for the predicted labels. (only required in case of multi-class)
    y_pred : array-like
        Predicted labels from model.
    normalize : bool, optional
        Normalize the values in the confusion matrix. The default is False.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    y_true : array-like with str
        True labels
    y_pred : array-like with str
        Prediction using (test)dataset, being class with pos_label or neg_label
    y_proba : array-like with float
        Probabilities of prediction on the (test)dataset
    class_names : dict
        False: neg_label, True: Positive
    ROCAUC : float
        Area under the curve
    stackbar : array of floats
        summarized information to make a multi-class bar-graph.
    confmat : dict containing keys
        Confusion-matrix, class_names and bool value whether the confusion matrix was normalized.

    """
    roc_scores = ROC.eval(y_true, y_proba, y_score, verbose=verbose)
    stackbar = _stackedbar_multiclass(y_true, y_pred, showfig=False)
    confmat = confmatrix.eval(y_true, y_pred, normalize=normalize)

    out = {}
    out['y_true'] = y_true
    out['y_pred'] = y_pred
    out['y_proba'] = y_proba
    out['class_names'] = np.unique(y_true)
    out['ROCAUC'] = roc_scores
    out['stackbar'] = stackbar
    out['confmat'] = confmat

    return(out)


# %% Two class results
def eval_twoclass(y_true, y_proba, pos_label=None, threshold=0.5, normalize=False, verbose=3):
    """Evaluate for two-class model.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    pos_label : str
        Positive label (only for the two-class models and when y_true is of type string. If you set bool, then the positive label is True)
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.
    normalize : bool, optional
        Normalize the values in the confusion matrix. The default is False.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    output is a dict containing the keys:

    class_names : dict
        False: neg_label, True: Positive
    pos_label : str
        Positive class label
    neg_label : str
        Negative class label (i.e., labels that are not positive)
    y_true : array-like with str
        True labels
    y_pred : array-like with str
        Prediction using (test)dataset, being class with pos_label or neg_label
    y_proba : array-like with float
        Probabilities of prediction on the (test)dataset
    auc : float
        Area under the curve
    f1 : float
        F1-score
    kappa : float
        Kappa-score
    report : str in table format
        A summary of precision, recall, f1 and support vs. macro/weighted accuracy
    thresholds : array of floats
        ROC scores
    fpr : array of float
        false positve rate
    tpr : array of float
        true positive rate
    average_precision : float
        Average precision
    precision : list of float
        The precision scores
    recall : list of float
        The recall scores
    MCC : float
        The MCC score
    CAP : float
        The CAP score
    TPFP : dict containing keys
        Indices of the FN, FP, TN, TP are listed.
    confmat : dict containing keys
        Confusion-matrix, class_names and bool value whether the confusion matrix was normalized.
    threshold : float
        Cut-off point to assign to a class

    """
    if (pos_label is None) and (str(y_true.dtype) != 'bool'):
        raise Exception('[classeval] CAP should have input argument <pos_label> or <y_true> being of type bool.')

    y_pred = y_proba>=threshold
    # Ensure y_label is a NumPy array of Python strings. This avoids errors when
    # y_true is a pandas Series with pandas.StringDtype which NumPy cannot
    # directly interpret as a dtype (TypeError). Converting to an ndarray of
    # Python str gives a stable NumPy-compatible dtype.
    y_label = np.asarray(y_true.astype(str))

    # if len(np.unique(y_true))>2:
    #    raise Exception('[classeval] This function is to evaluate two-class models and not multi-class.')
    if (pos_label is not None) and (y_true.dtype != 'bool'):
        # If y_true is strings, convert to bool based on positive label
        pos_label = str(pos_label)
        # y_true = y_label==pos_label
        y_true = np.isin(y_label, pos_label)
        neg_label = np.setdiff1d(np.unique(y_label),pos_label)[0]
        # Determine y_pred
        y_pred_label = np.repeat('',len(y_true)).astype(y_label.dtype)
        I = y_proba>=threshold
        y_pred_label[I] = pos_label
        y_pred_label[~I] = neg_label
    else:
        pos_label=True
        neg_label=False
        y_pred_label = y_pred.astype(str)

    # In sklearn, the output of the proba function has 2 columns.
    # The first column is the probability that the entry has the -1 label
    # The second column is the probability that the entry has the +1 label
    class_names = [neg_label, pos_label]

    # ROC plot
    ROC_scores = ROC.eval(y_true, y_proba, threshold=threshold, verbose=verbose)
    if verbose>=3: print('[classeval] AUC: %.2f' %(ROC_scores['auc']))
    # F1 score
    f1score = f1_score(y_true, y_pred)
    if verbose>=3: print('[classeval] F1: %.2f' %(f1score))
    # Classification report
    clreport = classification_report(y_true, (y_pred).astype(int))
    # Kappa score
    kappa_score = cohen_kappa_score(y_true, y_pred)
    if verbose>=3: print('[classeval] Kappa: %.2f' %(kappa_score))
    # Recall
    [precision, recall, _] = precision_recall_curve(y_true, y_proba)
    # MCC (Matthews Correlation Coefficient)
    outMCC = MCC(y_true, y_proba, threshold=threshold)
    if verbose>=3: print('[classeval] MCC score: %.2f' %(outMCC))
    # Average precision score
    average_precision = AP(y_true, y_proba, showfig=False)['AP']
    if verbose>=3: print('[classeval] Average precision (AP): %.2f' %(average_precision))
    # CAP plot
    CAP_score = CAP(y_true, y_proba, showfig=False)
    if verbose>=3: print('[classeval] CAP: %.0f' %(CAP_score))
    # Probability plot
    TPFP_scores = TPFP(y_true, y_proba, threshold=threshold, showfig=False)
    # Confusion matrix
    confmat = confmatrix.eval(y_label, y_pred_label, normalize=normalize, verbose=verbose)

    # Store and return
    out = {}
    out['class_names'] = class_names
    out['pos_label'] = pos_label
    out['neg_label'] = neg_label
    out['y_true'] = y_true
    out['y_pred'] = y_pred
    out['y_proba'] = y_proba
    out['auc'] = ROC_scores['auc']
    out['f1'] = f1score
    out['kappa'] = kappa_score
    out['report'] = clreport
    out['thresholds'] = ROC_scores['thresholds']
    out['fpr'] = ROC_scores['fpr']
    out['tpr'] = ROC_scores['tpr']
    out['average_precision'] = average_precision
    out['precision'] = precision
    out['recall'] = recall
    out['MCC'] = outMCC
    out['CAP'] = CAP_score
    out['TPFP'] = TPFP_scores
    out['confmat'] = confmat
    out['threshold'] = threshold

    return(out)


# %% MCC (Matthews Correlation Coefficient)
def MCC(y_true, y_proba, threshold=0.5):
    """MCC is extremely good metric for the imbalanced classification.

    Description
    -----------
    Score Ranges between [−1,1]
    1: perfect prediction
    0: random prediction
    −1: Total disagreement between predicted scores and true labels values.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.

    Returns
    -------
    MCC : float
        MCCscore.

    """
    y_true = (y_true).astype(int)
    y_pred = (y_proba>=threshold).astype(int)
    # MCC score
    MCC = matthews_corrcoef(y_true,y_pred)

    return(MCC)


# %% Creating probabilty classification plot
def AP(y_true, y_proba, title='', ax=None, figsize=(12, 8), fontsize=12, showfig=False):
    """AP (Average Precision) method.

    Description
    -----------
    A better metric in an imbalanced situation is the AUC PR (Area Under the Curve Precision Recall), or also called AP (Average Precision).
    If the precision decreases when we increase the recall, it shows that we have to choose a prediction thresold adapted to our needs.
    If our goal is to have a high recall, we should set a low prediction thresold that will allow us to detect most of the observations of the positive class,
    but with a low precision. On the contrary, if we want to be really confident about our predictions but don't mind about not finding all the positive observations,
    we should set a high thresold that will get us a high precision and a low recall.
    In order to know if our model performs better than another classifier, we can simply use the AP metric. To assess the quality of our model,
    we can compare it to a simple decision baseline. Let's take a random classifier as a baseline here that would predict half of the time 1 and half of the time 0 for the label.
    Such a classifier would have a precision of 4.3%, which corresponds to the proportion of positive observations.
    For every recall value the precision would stay the same, and this would lead us to an AP of 0.043.
    The AP of our model is approximately 0.35, which is more than 8 times higher than the AP of the random method.
    This means that our model has a good predictive power.

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

    Returns
    -------
    dict containing the following keys:

    AP : float
        Average precision score
    precision : list of float
        The precision scores
    recall : list of float
        The recall scores

    """
    average_precision = average_precision_score(y_true, y_proba)
    [precision, recall, _] = precision_recall_curve(y_true, y_proba)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    # Plot figure
    if showfig:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#F8F9FA')

        baseline = np.sum(np.asarray(y_true).astype(int)) / len(y_true)
        _color = '#3A86FF'

        ax.step(recall, precision, color=_color, lw=2.5, where='post', label=f'PR curve (AP = {average_precision:.2f})')
        ax.fill_between(recall, precision, alpha=0.15, color=_color, **step_kwargs)
        ax.axhline(baseline, color='#6C757D', linestyle='--', lw=1.5, label=f'Random classifier (AP = {baseline:.2f})')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall', fontsize=fontsize)
        ax.set_ylabel('Precision', fontsize=fontsize)
        _pr_title = f'Precision-Recall Curve — {title}' if title else 'Precision-Recall Curve'
        ax.set_title(_pr_title, fontsize=fontsize + 1, fontweight='bold', pad=12)
        ax.legend(loc='upper right', fontsize=fontsize - 1, framealpha=0.9)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    out = {}
    out['AP'] = average_precision
    out['precision'] = precision
    out['recall'] = recall
    return(out)


# %% Creating probabilty classification plot
def TPFP(y_true, y_proba, threshold=0.5, fontsize=12, title='', ax=None, figsize=(12, 8), showfig=False):
    """Plot the probabilties for both classes in a ordered manner.

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

    Returns
    -------
    dict containing the following keys FN, FP, TN, TP that contain the associated indices.

    """
    tmpout = pd.DataFrame()
    tmpout['pred_class'] = y_proba
    tmpout['true'] = y_true
    tmpout.sort_values(by=['true','pred_class'], ascending=False, inplace=True)
    tmpout.reset_index(drop=True, inplace=True)

    Itrue=tmpout['true'].values==1
    # True Positive class
    Itp=(tmpout['pred_class']>=threshold) & (Itrue)
    # True negative class
    Itn=(tmpout['pred_class']<threshold) & (Itrue)
    # False positives
    Ifp=(tmpout['pred_class']>=threshold) & (Itrue==False)
    # False negative class
    Ifn=(tmpout['pred_class']<threshold) & (Itrue==False)

    # Plot figure
    if showfig:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#F8F9FA')

        _style = dict(s=50, zorder=3, edgecolors='white', linewidths=0.6)
        ax.scatter(tmpout.index[Itp], tmpout['pred_class'].loc[Itp],
                   c='#2CA02C', marker='o', label='True Positive', **_style)
        ax.scatter(tmpout.index[Itn], tmpout['pred_class'].loc[Itn],
                   c='#98DF8A', marker='X', label='True Negative', **_style)
        ax.scatter(tmpout.index[Ifp], tmpout['pred_class'].loc[Ifp],
                   c='#FF7F7F', marker='X', label='False Positive', **_style)
        ax.scatter(tmpout.index[Ifn], tmpout['pred_class'].loc[Ifn],
                   c='#D62728', marker='o', label='False Negative', **_style)

        ax.axhline(threshold, color='#E87722', linestyle='--', lw=1.8,
                   label=f'Threshold = {threshold:.2f}')
        ax.set_ylim([-0.08, 1.08])
        ax.set_ylabel('P(positive class | X)', fontsize=fontsize)
        ax.set_xlabel('Sample index', fontsize=fontsize)
        _tpfp_title = f'Prediction Probabilities — {title}' if title else 'Prediction Probabilities'
        ax.set_title(_tpfp_title, fontsize=fontsize + 1, fontweight='bold', pad=12)
        ax.legend(fontsize=fontsize - 1, framealpha=0.9,
                  loc='upper right', ncol=2)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    out = {}
    out['TP'] = np.where(Itp)[0]
    out['TN'] = np.where(Itn)[0]
    out['FP'] = np.where(Ifp)[0]
    out['FN'] = np.where(Ifn)[0]

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
    dict containing the following keys:

    macro_roc_auc_ovo : float
        AUC score based on One-vs-One scheme
    weighted_roc_auc_ovo : float
        Weighted AUC score based on One-vs-One scheme
    macro_roc_auc_ovr : float
        AUC score based on One-vs-Rest scheme
    weighted_roc_auc_ovr : float
        Weighted AUC score based on One-vs-Rest scheme

    """

    macro_roc_auc_ovo = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_true, y_proba, multi_class="ovo", average="weighted")
    macro_roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    if verbose>=3:
        print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} " "(weighted by prevalence)" .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
        print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} " "(weighted by prevalence)" .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    out = {}
    out['macro_roc_auc_ovo'] = macro_roc_auc_ovo
    out['weighted_roc_auc_ovo'] = weighted_roc_auc_ovo
    out['macro_roc_auc_ovr'] = macro_roc_auc_ovr
    out['weighted_roc_auc_ovr'] = weighted_roc_auc_ovr
    return(out)


# %% CAP
def CAP(y_true, y_pred, label='Classifier', ax=None, figsize=(12, 8), fontsize=12, showfig=False):
    """Compute Cumulitive Accuracy Profile (CAP) to measure the performance in a two class classifier.

    Description
    -----------
    The CAP Curve analyse how to effectively identify all data points of a given class using minimum number of tries.
    This function computes Cumulitive Accuracy Profile (CAP) to measure the performance of a classifier.
    It ranks the predicted class probabilities (high to low), together with the true values.
    With that, it computes the cumsum which is the final line.
    A perfect model is one which will detect all class 1.0 data points in the same number of tries as there are class 1.0 data points.


    Parameters
    ----------
    y_true : array-like [list or numpy]
        list of true labels.
    y_pred : array-like [list or numpy]
        list of predicted labels.
    label : str, optional
        Label. The default is 'Classifier'.
    ax : figure object, optional
        provide an existing axis to make the plot. The default is None.
    fontsize : int, optional
        Fontsize in the figures. The default is 12.
    showfig : bool, optional
        Show figure. The default is True.

    Returns
    -------
    float : CAP score.

    """
    total = len(y_true)
    # Probs and y_test are zipped together.
    # Sort this zip in the reverse order of probabilities such that the maximum probability comes first and then lower probabilities follow.
    # I extract only the y_test values in an array and store it in model_y.
    prob_and_true = sorted(zip(y_pred, y_true), reverse=True)
    model_y = [y for _, y in prob_and_true]
    # creates an array of values while cumulatively adding all previous values in the array to the present value.
    y_values = np.append([0], np.cumsum(model_y))
    x_values = np.arange(0, total + 1)
    CAP_score = max(y_values)

    if showfig:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#F8F9FA')
        class_1_count = np.sum(y_true)

        # Filled area between model and random
        ax.fill_between(x_values, y_values, np.linspace(0, class_1_count, len(x_values)),
                        alpha=0.12, color='#E87722')

        ax.plot([0, total], [0, class_1_count], c='#6C757D', linestyle='--', lw=1.5, label='Random model')
        ax.plot([0, class_1_count, total], [0, class_1_count, class_1_count],
                c='#2CA02C', linestyle='-.', lw=1.5, label='Perfect model')
        ax.plot(x_values, y_values, c='#E87722', label=label, linewidth=2.5, zorder=3)

        ax.legend(loc='lower right', fontsize=fontsize - 1, framealpha=0.9)
        ax.set_xlabel('Total observations', fontsize=fontsize)
        ax.set_ylabel('Class observations', fontsize=fontsize)
        ax.set_title(f'Cumulative Accuracy Profile (CAP)  |  Score: {CAP_score}',
                     fontsize=fontsize + 1, fontweight='bold', pad=12)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return(CAP_score)


# %% Import example dataset from github.
def load_example(data='breast'):
    """Import example dataset from sklearn.

    Parameters
    ----------
    'breast' : str, two-class
    'titanic': str, two-class
    'iris' : str, multi-class

    Returns
    -------
    tuple containing dataset and response variable (X,y).

    """
    try:
        from sklearn import datasets
    except:
        print('This requires: <pip install sklearn>')
        return None, None

    if data=='iris':
        iris = datasets.load_iris()
        X = pd.DataFrame(index=iris['target_names'][iris['target']], data=iris['data'], columns=iris['feature_names'])
        y = X.index.values
    elif data=='breast':
        breast = datasets.load_breast_cancer()
        X = pd.DataFrame(index=breast['target_names'][breast['target']], data=breast['data'], columns=breast['feature_names'])
        y = X.index.values
    elif data=='titanic':
        X, y = datasets.fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        y=y.astype(str)
        y[y=='1']='survived'
        y[y=='0']='dead'
        X.index = y
    return X, y


# %% Two class results
def _stackedbar_multiclass(y_true, y_pred, fontsize=12, showfig=False, bar_edgecolor='black'):
    uiy = np.unique(y_true)
    df = pd.DataFrame(data=np.zeros((len(uiy),len(uiy))), index=uiy, columns=uiy)
    for y in uiy:
        I = y_true==y
        labels, n = np.unique(y_pred[I], return_counts=True)
        # Use single-step .loc assignment to avoid pandas SettingWithCopyWarning.
        df.loc[labels, y] = n

    if showfig:
        palette = plt.cm.Set2(np.linspace(0, 0.85, len(uiy)))
        ax_bar = df.T.plot(kind='bar', stacked=True, color=palette,
                           figsize=(max(8, len(uiy) * 2), 6), edgecolor=bar_edgecolor, linewidth=0.8)
        fig_bar = ax_bar.get_figure()
        fig_bar.patch.set_facecolor('#F8F9FA')
        ax_bar.set_facecolor('#F8F9FA')

        # Annotate each segment with its count (skip zeros)
        for container in ax_bar.containers:
            for bar in container:
                h = bar.get_height()
                if h > 0:
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + h / 2,
                        f'{int(h)}',
                        ha='center', va='center',
                        fontsize=fontsize - 2, color='white', fontweight='bold'
                    )

        ax_bar.set_ylabel('Number of predictions', fontsize=fontsize)
        ax_bar.set_xlabel('True class', fontsize=fontsize)
        ax_bar.set_title('Predicted class distribution per true class',
                         fontsize=fontsize + 1, fontweight='bold', pad=12)
        ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=30, ha='right', fontsize=fontsize - 1)
        ax_bar.legend(title='Predicted class', fontsize=fontsize - 2,
                      title_fontsize=fontsize - 2, framealpha=0.9)
        ax_bar.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        fig_bar.tight_layout()

    return df
