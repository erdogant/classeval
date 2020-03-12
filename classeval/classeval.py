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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score, precision_recall_curve
from funcsigs import signature
import classeval.confmatrix as confmatrix


# %% Main function for all two class results.
def summary(y_true, y_proba, threshold=0.5, title='', showfig=True, verbose=3):
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

    if showfig:
        [fig, ax] = plt.subplots(2,2,figsize=(28,16))
    else:
        ax = {}
        ax[0] = [None,None]
        ax[1] = [None,None]

    # Create classification report
    out = two_class(y_true, y_proba, threshold=threshold, verbose=verbose)
    # pr curve
    AP(y_true, y_proba, title=title, ax=ax[1][0], showfig=showfig)
    # ROC plot
    _ = ROC(y_true, y_proba, threshold=threshold, title=title, ax=ax[0][0], showfig=showfig, verbose=0)
    # CAP plot
    out['CAP'] = CAP(y_true, y_proba, ax=ax[0][1], showfig=showfig)
    # Probability plot
    out['TPFP'] = proba_curve(y_true, y_proba, threshold=threshold, title=title, ax=ax[1][1], showfig=showfig)
    # Show plot
    if showfig: plt.show()

    # Confusion matrix
    out['confmat'] = confmatrix.eval(y_true, (y_proba>=threshold).astype(int), verbose=verbose)
    if showfig:
        _ = confmatrix.plot(out['confmat'], title=title, cmap=plt.cm.Blues, figsize=(8,8))
    # Return
    return(out)


# %% Two class results
def two_class(y_true, y_proba, threshold=0.5, verbose=3):
    """Evaluate for two-class model.

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
    dict containing results.

    """
    if len(np.unique(y_true))>2:
        raise Exception('[classeval] This function is to evaluate two-class models and not multi-class.')
    # ROC curve
    [fpr, tpr, thresholds] = roc_curve(y_true, y_proba)
    # AUC
    roc_auc = auc(fpr, tpr)
    if verbose>=3: print('[classeval] AUC: %.2f' %(roc_auc))
    # F1 score
    f1score = f1_score(y_true, y_proba>=threshold)
    if verbose>=3: print('[classeval] F1: %.2f' %(f1score))
    # Classification report
    clreport = classification_report(y_true, (y_proba>=threshold).astype(int))
    # Kappa score
    kappscore = cohen_kappa_score(y_true, y_proba>=threshold)
    if verbose>=3: print('[classeval] Kappa: %.2f' %(kappscore))
    # Average precision score
    average_precision = average_precision_score(y_true, y_proba)
    # Recall
    [precision, recall, _] = precision_recall_curve(y_true, y_proba)
    # MCC (Matthews Correlation Coefficient)
    outMCC = MCC(y_true, y_proba, threshold=threshold)

    # Store and return
    out = {}
    out['auc'] = roc_auc
    out['f1'] = f1score
    out['kappa'] = kappscore
    out['report'] = clreport
    out['thresholds'] = thresholds
    out['fpr'] = fpr
    out['tpr'] = tpr
    out['average_precision'] = average_precision
    out['precision'] = precision
    out['recall'] = recall
    out['MCC'] = outMCC

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
def AP(y_true, y_proba, title='', ax=None, showfig=True):
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
            [fig,ax]= plt.subplots(figsize=(15,8))

        ax.step(recall, precision, color='b', alpha=0.2, where='post')
        ax.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        ax.grid(True)

    out = {}
    out['AP'] = average_precision
    out['precision'] = precision
    out['recall'] = recall
    return(out)


# %% Creating probabilty classification plot
def proba_curve(y_true, y_proba, threshold=0.5, title='', ax=None, showfig=True):
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
        if isinstance(ax,type(None)): [fig,ax]= plt.subplots(figsize=(20,8))
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
        ax.set_ylabel('P(class | X)')
        ax.set_xlabel('Samples')
        ax.grid(True)
        ax.legend()
        plt.show()

    out = {}
    out['TP'] = np.where(Itp)[0]
    out['TN'] = np.where(Itn)[0]
    out['FP'] = np.where(Ifp)[0]
    out['FN'] = np.where(Ifn)[0]

    return(out)


# %% ROC plot
def ROC(y_true, y_proba, threshold=0.5, title='', ax=None, showfig=True, verbose=3):
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
    verbose : int, optional
        print message to screen. The default is 3.

    Returns
    -------
    dict containing results.

    """
    # Create classification report
    out=two_class(y_true, y_proba, threshold=threshold, verbose=verbose)

    # Plot figure
    if showfig:
        if isinstance(ax,type(None)):
            [fig,ax]= plt.subplots(figsize=(12,8))

        lw = 2
        ax.plot(out['fpr'], out['tpr'], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % out['auc'])
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('[%s] Receiver operating characteristic. F1:%.3f, Kappa:%.3f' %(title, out['f1'], out['kappa']))
        ax.legend(loc="lower right")
        ax.grid(True)

    return(out)


# %% CAP
def CAP(y_true, y_pred, label='Classifier', ax=None, showfig=True):
    """Compute Cumulitive Accuracy Profile (CAP) to measure the performance in a two class classifier.

    Description
    -----------
    The CAP Curve analyse how to effectively identify all data points of a given class using minimum number of tries.
    This function computes Cumulitive Accuracy Profile (CAP) to measure the performance of a classifier.
    It ranks the predicted class probabilities (high to low), together with the true values.
    With that, it computes the cumsum which is the final line.

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
    showfig : bool, optional
        Show figure. The default is True.

    Returns
    -------
    float : CAP score.

    """
    config = dict()
    config['label']=label
    fontsize=14

    total = len(y_true)
    class_1_count = np.sum(y_true)

    if showfig:
        if ax is None: fig,ax = plt.subplots(figsize=(20, 12))
        ax.plot([0, total], [0, class_1_count], c='navy', linestyle='--', label='Random Model')
        ax.set_xlabel('Total observations', fontsize=fontsize)
        ax.set_ylabel('Class observations', fontsize=fontsize)
        ax.set_title('Cumulitive Accuracy Profile (CAP)', fontsize=fontsize)
        ax.grid(True)

        # A perfect model is one which will detect all class 1.0 data points in the same number of tries as there are class 1.0 data points.
        # It takes exactly 58 tries for the perfect model to identify 58 class 1.0 data points.
        ax.plot([0, class_1_count, total], [0, class_1_count, class_1_count], c='grey', linewidth=1, label='Perfect Model')

    # Probs and y_test are zipped together.
    # Sort this zip in the reverse order of probabilities such that the maximum probability comes first and then lower probabilities follow.
    # I extract only the y_test values in an array and store it in model_y.
    prob_and_true = sorted(zip(y_pred, y_true), reverse=True)
    model_y = [y for _, y in prob_and_true]

    # creates an array of values while cumulatively adding all previous values in the array to the present value.
    y_values = np.append([0], np.cumsum(model_y))
    x_values = np.arange(0, total + 1)

    # Plot accuracy
    if showfig:
        ax.plot(x_values, y_values, c='darkorange', label=config['label'], linewidth=2)
        ax.legend(loc='lower right', fontsize=fontsize)

    return(max(y_values))


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
        X, y = datasets.load_iris(return_X_y=True)
    elif data=='breast':
        X, y = datasets.load_breast_cancer(return_X_y=True)
    elif data=='titanic':
        X, y = datasets.fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    return X, y
