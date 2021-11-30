# classeval

[![Python](https://img.shields.io/pypi/pyversions/classeval)](https://img.shields.io/pypi/pyversions/classeval)
[![PyPI Version](https://img.shields.io/pypi/v/classeval)](https://pypi.org/project/classeval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/classeval/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/classeval)](https://pepy.tech/project/classeval)
[![Downloads](https://pepy.tech/badge/classeval/month)](https://pepy.tech/project/classeval/month)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/classeval/)
[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
[![DOI](https://zenodo.org/badge/246504758.svg)](https://zenodo.org/badge/latestdoi/246504758)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

The library ``classeval`` is developed to evaluate the models performance of any kind of **two-class** or **multi-class** model. ``classeval`` computes many scoring measures in case of a two-class clasification model. Some measures are utilized from ``sklearn``, among them AUC, MCC, Cohen kappa score, matthews correlation coefficient, whereas others are custom. This library can help to consistenly compare the output of various models. In addition, it can also give insights in tuning the models performance as the the threshold being used can be adjusted and evaluated. The output of ``classeval`` can subsequently plotted in terms of ROC curves, confusion matrices, class distributions, and probability plots. Such plots can help in better understanding of the results.

### Docs
Navigate to [API documentations](https://erdogant.github.io/classeval/) for more detailed information.



### Contents
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install classeval from PyPI (recommended). classeval is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

#### Quick Start
```
pip install classeval
```

* Alternatively, install classeval from the GitHub source:
```bash
git clone https://github.com/erdogant/classeval.git
cd classeval
python setup.py install
```  

#### Import classeval package
```python
import classeval as clf
```

#### Example two-class model:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
gb = GradientBoostingClassifier()

X, y = clf.load_example('breast')
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2)
model = gb.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

# Evaluate
out = clf.eval(y_true, y_proba, pos_label='malignant')
ax = clf.plot(out, figsize=(20,15), fontsize=14)
```

<p align="center">
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/Figure_1.png" width="600" />
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/Figure_2.png" width="400" />
</p>

#### Example multi-class model:
```python

X,y = clf.load_example('iris')
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.5)

model = gb.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_score = model.decision_function(X_test)

# All
out = clf.eval(y_true, y_proba, y_score, y_pred)
ax = clf.plot(out)
```

#### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

#### Contribute
* Contributions are welcome.
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

#### Licence
* See [LICENSE](LICENSE) for details.
