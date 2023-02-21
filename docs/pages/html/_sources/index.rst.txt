
Classeval's documentation!
==========================

|python| |pypi| |docs| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |DOI| |donate|


.. |figIP1| image:: ../figs/example1_fig1.png

.. table:: 
   :align: center

   +----------+
   | |figIP1| |
   +----------+


-----------------------------------


The library ``classeval`` is developed to evaluate the models performance of any kind of **two-class** or **multi-class** model. ``classeval`` computes many scoring measures in case of a two-class clasification model. Some measures are utilized from ``sklearn``, among them AUC, MCC, Cohen kappa score, matthews correlation coefficient, whereas others are custom. This library can help to consistenly compare the output of various models. In addition, it can also give insights in tuning the models performance as the the threshold being used can be adjusted and evaluated. The output of ``classeval`` can subsequently plotted in terms of ROC curves, confusion matrices, class distributions, and probability plots. Such plots can help in better understanding of the results.


-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can support in various ways, have a look at the `sponser page <https://erdogant.github.io/classeval/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/classeval/>`_ page.

	.. code-block:: console

	   pip install classeval

-----------------------------------

Contents
========


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation


.. toctree::
  :maxdepth: 1
  :caption: Methods

  Methods


.. toctree::
  :maxdepth: 1
  :caption: Plots

  Plots


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 1
  :caption: Documentation

  Documentation
  Coding quality
  classeval.classeval

* :ref:`genindex`



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. |python| image:: https://img.shields.io/pypi/pyversions/classeval.svg
    :alt: |Python
    :target: https://erdogant.github.io/classeval/

.. |pypi| image:: https://img.shields.io/pypi/v/classeval.svg
    :alt: |Python Version
    :target: https://pypi.org/project/classeval/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/classeval/

.. |LOC| image:: https://sloc.xyz/github/erdogant/classeval/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/classeval

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/classeval?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/classeval

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/classeval?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/classeval

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/classeval/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/classeval.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/classeval/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/classeval.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/classeval/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/classeval/pages/html/Documentation.html#

.. |DOI| image:: https://zenodo.org/badge/246504758.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/246504758


.. include:: add_bottom.add