
Eval
''''''''

The ``classeval`` library contains various functions to plot the results. The main function to plot is the function :func:`classeval.classeval.plot`. This function automatically determines whether the trained model is based on a two-class or multi-class approach. Plotting results is possible by simply using the :func:`classeval.classeval.plot` functionality.


Lets first train a simple model:

.. code:: python

    # Load data
    X, y = clf.load_example('breast')
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2)

    # Fit model
    model = gb.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)


Evaluate the models performance and use the output for making the plots.

.. code:: python

    # Import library
    import classeval as clf

    # Evaluate model
    out = clf.eval(y_true, y_proba, pos_label='malignant')

    # Make plot
    ax = clf.plot(out)


**Two-class**

.. example1_fig1:

.. figure:: ../figs/example1_fig1.png



ROC-AUC
''''''''''''

Plotting the ROC is initially desiged for two-class models. However, with some adjustments it is also possible to plot the ROC for a multi-class model under the assumption that is OvR or OvO schemes. See methods section for more details. With the function :func:`classeval.ROC.eval` the ROC and AUC can be examined and plotted for both the two-class as well as multi-class models.

.. code:: python

    # Compute ROC
    outRoc = clf.ROC.eval(y_true, y_proba, pos_label='malignant')

    # Make plot
    ax = clf.ROC.plot(outRoc)


**Two-class**

.. ROC_twoclass:

.. figure:: ../figs/ROC_twoclass.png
    :scale: 80%


**Multi-class**

.. multiclass_fig1_1:

.. figure:: ../figs/multiclass_fig1_1.png
    :scale: 80%



CAP
''''''''

.. code:: python

    # Compute CAP
    outCAP = clf.CAP(out['y_true'], out['y_proba'], showfig=True)


.. CAP_fig:

.. figure:: ../figs/CAP_fig.png
    :scale: 80%



AP 
'''''''''''''''''

.. code:: python

    # Compute AP
    outAP = clf.AP(out['y_true'], out['y_proba'], showfig=True)


.. AP_fig:

.. figure:: ../figs/AP_fig.png
    :scale: 80%



Confusion matrix 
'''''''''''''''''''''

A confusion matrix is a table to describe the performance of a classification model.
With the function :func:`classeval.confmatrix.eval` the confusion matrix can be examined and plotted for both the two-class as well as multi-class model.


.. code:: python

    # Compute confmatrix
    outCONF = clf.confmatrix.eval(y_true, y_pred)

    # Make plot
    ax = clf.confmatrix.plot(outCONF)


**Two-class**

.. Figure_2:

.. figure:: ../figs/Figure_2.png
    :scale: 50%


**Multi-class**

.. multiclass_fig1_4:

.. figure:: ../figs/multiclass_fig1_4.png
    :scale: 50%




Probability Plot
''''''''''''''''''

The probability plot depicts the probabilities of the samples being classified.
This function is desiged for only two-class models and callable via: :func:`classeval.classeval.TPFP`


.. code:: python

    # Compute TPFP
    out = clf.TPFP(out['y_true'], out['y_proba'], showfig=True)


.. multiclass_threshold_05:

.. figure:: ../figs/multiclass_threshold_05.png
    :scale: 90%



.. include:: add_bottom.add