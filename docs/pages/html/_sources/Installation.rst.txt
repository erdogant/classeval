.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to evaluate the results of a trained model.


.. code:: python

    # Import library
    import classeval

    # Train model and derive probabilities
    X, y = clf.load_example('breast')
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2)
    model = gb.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    # Evaluate the models performance
    out = clf.eval(y_true, y_proba, pos_label='malignant')

    # Plot results
    ax = clf.plot(out)



Installation
''''''''''''

Install via ``pip``:

.. code-block:: console

    # Installation from pypi
    pip install classeval


Install via github source:

.. code-block:: console

    # Install directly from github
    pip install git+https://github.com/erdogant/classeval
