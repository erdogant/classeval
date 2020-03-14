from classeval.classeval import (
    eval,
    plot,
    load_example,
)

import classeval.confmatrix as confmatrix
import classeval.ROC as ROC

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.3'

# module level doc-string
__doc__ = """
classeval
=====================================================================

Description
-----------
classeval is a python package that contains functionalities and plots for fast and easy classifier evaluation.


Example
-------
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
>>> gb = GradientBoostingClassifier()
>>>
>>> import classeval as clf
>>>
>>> X, y = clf.load_example('breast')
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
>>>
>>> model = gb.fit(X_train, y_train)
>>> y_proba = model.predict_proba(X_test)
>>> y_pred = model.predict(X_test)
>>>
>>> results = clf.eval(y_test, y_proba[:,1])
>>> print(results['report'])


References
----------
* https://github.com/erdogant/classeval
* https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
* http://arogozhnikov.github.io/2015/10/05/roc-curve.html
* https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
* https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b


"""
