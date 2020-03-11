from classeval.classeval import (
    summary,
    two_class,
    MCC,
    pr_curve,
    proba,
    ROC,
    CAP,
    load_example,
)

import classeval.confmatrix as confmatrix

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

# module level doc-string
__doc__ = """
classeval
=====================================================================

Description
-----------
classeval is a python package that contains functionalities and plots for fast and easy classifier evaluation.

Example
-------
>>> import classeval as clf
>>> results = clf.eval(model)
>>> fig,ax = clf.plot(model)

References
----------
* https://github.com/erdogant/classeval
* https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
* http://arogozhnikov.github.io/2015/10/05/roc-curve.html


"""
