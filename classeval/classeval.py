"""classeval contains functionalities and plots for fast and easy classifier evaluation."""
# -----------------------------------------------------
# Name        : classeval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/classeval
# Licence     : MIT
# -----------------------------------------------------

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
import classeval.confmatrix as confmatrix
import classeval.ROC as ROC


# %% Main function for all two class results.
def plot(out, title='', fontsize=12, figsize=(20,15)):

    y_true = out['y_true']
    y_proba = out['y_proba']
    threshold = out['threshold']

    if len(out['class_names'])==2:
        # Setup figure
        [fig, ax] = plt.subplots(2,2, figsize=figsize)
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
        _ = confmatrix.plot(out['confmat'], title=title, cmap=plt.cm.Blues)
    elif len(out['class_names'])>2:
        # Setup figure
        # [fig, ax] = plt.subplots(2,1,figsize=figsize)
        ax = None
        _ = ROC.plot(out['ROCAUC'], fontsize=fontsize)
        _ = confmatrix.plot(out['confmat'])
        _ = _stackedbar_multiclass(out['y_true'], out['y_pred'], showfig=True, fontsize=fontsize)

    return ax


# %% Main function for all two class results.
def eval(y_true, y_proba, y_score=None, y_pred=None, pos_label=None, threshold=0.5, verbose=3):
    """Evaluate and make plots for two-class models.

    Parameters
    ----------
    y_true : array-like [list or int]
        True labels of the classes.
    y_proba : array of floats
        Probabilities of the predicted labels.
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.
    title : str, optional
        Title of the plot. The default is ''.
    showfig : bool, optional
        Show the figure to screen. The default is True.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing multiple results.

    """
    if isinstance(y_proba, pd.DataFrame):
        print('[classeval] pandas DataFrame not allowed as input for y_proba. Use lists or numpy array-like')
    if isinstance(y_true, pd.DataFrame):
        print('[classeval] pandas DataFrame not allowed as input for y_true. Use lists or numpy array-like')
    if pos_label is not None:
        if not np.any(y_true==pos_label):
            raise Exception(['[classeval] pos_label is not found in y_true!'])

    # Create classification report
    class_names = np.unique(y_true)
    if len(class_names)==2:
        out = eval_twoclass(y_true, y_proba, threshold=threshold, pos_label=pos_label, verbose=verbose)
    elif len(class_names)>2:
        out = eval_multiclass(y_true, y_proba, y_score, y_pred, threshold=threshold, verbose=verbose)
    else:
        out=None

    # Return
    return(out)


# %% Two class results
def _stackedbar_multiclass(y_true, y_pred, fontsize=12, showfig=False, verbose=3):
    uiy = np.unique(y_true)
    df = pd.DataFrame(data=np.zeros((3,3)), index=uiy, columns=uiy)
    for i, y in enumerate(uiy):
        I = y_true==y
        labels, n = np.unique(y_pred[I], return_counts=True)
        df[y].loc[labels]=n

    if showfig:
        df.plot(kind='bar', stacked=True)
        plt.ylabel('Number of predicted classes', fontsize=fontsize)
        plt.xlabel('True class', fontsize=12)
        plt.title('Class prediction', fontsize=12)

    return(df)


# %% Two class results
def eval_multiclass(y_true, y_proba, y_score, y_pred, threshold=0.5, normalize=False, verbose=3):
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
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.
    normalize : bool, optional
        Normalize the values in the confusion matrix. The default is False.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    """
    roc_scores = ROC.eval(y_true, y_proba, y_score, verbose=verbose)
    stackbar = _stackedbar_multiclass(y_true, y_pred, showfig=False, verbose=verbose)
    confmat = confmatrix.eval(y_true, y_pred, normalize=normalize)

    out = {}
    out['y_true'] = y_true
    out['y_pred'] = y_pred
    out['y_proba'] = y_proba
    out['threshold'] = threshold
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
        Positive label (only for the two-class models and when y_ture is of type string. If you set bool, then the positive label is True)
    threshold : float [0-1], optional
        Cut-off point to define the class label. The default is 0.5 in a two-class model.
    normalize : bool, optional
        Normalize the values in the confusion matrix. The default is False.
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    """
    class_names = np.unique(y_true)
    y_label = y_true.astype(str)
    y_pred = y_proba>=threshold

    if len(class_names)>2:
        raise Exception('[classeval] This function is to evaluate two-class models and not multi-class.')
    if (pos_label is None) and (y_true.dtype!='bool'):
        raise Exception('[classeval] CAP should have input argument <pos_label> or <y_true> being of type bool.')
    if (pos_label is not None) and (y_true.dtype!='bool'):
        # If y_true is strings, convert to bool based on positive label
        y_true = y_label==pos_label
        neg_label = np.setdiff1d(class_names,pos_label)[0]
        # Determine y_pred
        y_pred_label = np.repeat('',len(y_true)).astype(y_label.dtype)
        I = y_proba>=threshold
        y_pred_label[I] = pos_label
        y_pred_label[I==False] = neg_label
    else:
        pos_label=True
        neg_label=False
        y_pred_label = y_pred.astype(str)

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
def MCC(y_true, y_proba, threshold=0.5, verbose=3):
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
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    float containing mcc score.

    """
    y_true = (y_true).astype(int)
    y_pred = (y_proba>=threshold).astype(int)
    # MCC score
    MCC = matthews_corrcoef(y_true,y_pred)

    return(MCC)


# %% Creating probabilty classification plot
def AP(y_true, y_proba, title='', ax=None, figsize=(12,8), fontsize=12, showfig=False):
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
    For every recall value the precision would stay the same, and this would lead us to an AP of 0.043. The AP of our model is approximately 0.35, which is more than 8 times higher than the AP of the random method.
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
    dict containing results.

    """
    average_precision = average_precision_score(y_true, y_proba)
    [precision, recall, _] = precision_recall_curve(y_true, y_proba)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    # Plot figure
    if showfig:
        if isinstance(ax,type(None)):
            [fig,ax]= plt.subplots(figsize=figsize)

        ax.step(recall, precision, color='b', alpha=0.2, where='post')
        ax.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall', fontsize=fontsize)
        ax.set_ylabel('Precision', fontsize=fontsize)
        ax.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision), fontsize=fontsize)
        ax.grid(True)

    out = {}
    out['AP'] = average_precision
    out['precision'] = precision
    out['recall'] = recall
    return(out)


# %% Creating probabilty classification plot
def TPFP(y_true, y_proba, threshold=0.5, fontsize=12, title='', ax=None, figsize=(12,8), showfig=False):
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
    dict containing results.

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
        if isinstance(ax,type(None)): [fig,ax]= plt.subplots(figsize=figsize)
        # True Positive class
        ax.plot(tmpout['pred_class'].loc[Itp], 'g.',label='True Positive')
        # True negative class
        ax.plot(tmpout['pred_class'].loc[Itn], 'gx',label='True negative')
        # False positives
        ax.plot(tmpout['pred_class'].loc[Ifp], 'rx',label='False positive')
        # False negative class
        ax.plot(tmpout['pred_class'].loc[Ifn], 'r.',label='False negative')
        # Styling
        ax.hlines(threshold, 0,len(Itrue), 'r', linestyles='dashed')
        ax.set_ylim([-0.1, 1.1])
        ax.set_ylabel('P(class | X)', fontsize=fontsize)
        ax.set_xlabel('Samples', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.grid(True)
        plt.show()

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
    dict containing results.

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
def CAP(y_true, y_pred, pos_label=None, label='Classifier', ax=None, figsize=(12,8), fontsize=12, showfig=False):
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
    # if (pos_label is None) and (y_true.dtype=='O'):
    #     raise Exception('[classeval] CAP should have [pos_label] or [y_true] being bool.')
    # if (pos_label is not None) and (y_true.dtype=='O'):
    #     y_label = y_true
    #     y_true = y_label==pos_label

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
        # Setup figure
        if ax is None: fig,ax = plt.subplots(figsize=figsize)
        class_1_count = np.sum(y_true)
        ax.plot([0, total], [0, class_1_count], c='navy', linestyle='--', label='Random Model')
        ax.plot([0, class_1_count, total], [0, class_1_count, class_1_count], c='grey', linewidth=1, label='Perfect Model')
        # Plot accuracy
        ax.plot(x_values, y_values, c='darkorange', label=label, linewidth=2)
        # Set legends
        ax.legend(loc='lower right', fontsize=fontsize)
        ax.set_xlabel('Total observations', fontsize=fontsize)
        ax.set_ylabel('Class observations', fontsize=fontsize)
        ax.set_title(('Cumulitive Accuracy Profile (CAP), score: %s' %(CAP_score)), fontsize=fontsize)
        ax.grid(True)

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
