.. include:: add_top.add

Coding quality
'''''''''''''''''''''

We value software quality. Higher quality software has fewer defects, better security, and better performance, which leads to happier users who can work more effectively. The ``classeval`` library is developed with several techniques, such as coding styling, low complexity, docstrings, reviews, and unit tests. Such conventions are helpfull to improve the quality, make the code cleaner and more understandable but alos to trace future bugs, and spot syntax errors.


library
-------

The file structure of the generated package looks like:


.. code-block:: bash

    path/to/classeval/
    ├── .editorconfig
    ├── .gitignore
    ├── .pre-commit-config.yml
    ├── .prospector.yml
    ├── CHANGELOG.rst
    ├── docs
    │   ├── conf.py
    │   ├── index.rst
    │   └── ...
    ├── LICENSE
    ├── MANIFEST.in
    ├── NOTICE
    ├── classeval
    │   ├── __init__.py
    │   ├── __version__.py
    │   └── classeval.py
    ├── README.md
    ├── requirements.txt
    ├── setup.cfg
    ├── setup.py
    └── tests
        ├── __init__.py
        └── test_classeval.py


Style
-----

This library is compliant with the PEP-8 standards.
PEP stands for Python Enhancement Proposal and sets a baseline for the readability of Python code.
Each public function contains a docstring that is based on numpy standards.
    

Complexity
----------

This library has been developed by using measures that help decreasing technical debt.
Version 0.1.4 of the ``classeval`` library scored, according the code analyzer: **3.89**, for which values > 0 are good and 10 is a maximum score.
Developing software with low(er) technical dept may take extra development time, but has many advantages:

* Higher quality code
* Easier maintanable
* Less prone to bugs and errors
* Improved security


Unit tests
----------

The use of unit tests is essential to garantee a consistent output of developed functions.
The following tests are secured using :func:`tests.test_classeval`:

* The input are checked.
* The output values are checked and whether they are encoded properly.
* The check of whether parameters are handled correctly.


.. code-block:: bash

	pytest tests\test_classeval.py

	====================================== test session starts ======================================
	platform win32 -- Python 3.6.10, pytest-5.4.0, py-1.8.1, pluggy-0.13.1
	collected 1 item

	tests\test_classeval.py .

	================================ 1 passed, 3 warnings in 16.00s =================================



.. include:: add_bottom.add