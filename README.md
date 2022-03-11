# classeval

[![Python](https://img.shields.io/pypi/pyversions/classeval)](https://img.shields.io/pypi/pyversions/classeval)
[![PyPI Version](https://img.shields.io/pypi/v/classeval)](https://pypi.org/project/classeval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/classeval/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/classeval)](https://pepy.tech/project/classeval)
[![Downloads](https://pepy.tech/badge/classeval/month)](https://pepy.tech/project/classeval/month)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/classeval/)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

The library ``classeval`` is developed to evaluate the models performance of any kind of **two-class** or **multi-class** model. ``classeval`` computes many scoring measures in case of a two-class clasification model. Some measures are utilized from ``sklearn``, among them AUC, MCC, Cohen kappa score, matthews correlation coefficient, whereas others are custom. This library can help to consistenly compare the output of various models. In addition, it can also give insights in tuning the models performance as the the threshold being used can be adjusted and evaluated. The output of ``classeval`` can subsequently plotted in terms of ROC curves, confusion matrices, class distributions, and probability plots. Such plots can help in better understanding of the results.

# 
**⭐️ Star this repo if you like it ⭐️**
# 


### [Documentation pages](https://erdogant.github.io/classeval/)

On the [documentation pages](https://erdogant.github.io/classeval/) you can find more information about ``classeval`` with examples. 

# 

##### Install classeval from PyPI
```bash
pip install classeval     # normal install
pip install -U classeval  # update if needed
```


### Import classeval package
```python
import classeval as clf

```

<hr>

### Examples

# 

#### [Example: Evaluate Two-class model](https://erdogant.github.io/classeval/pages/html/Examples.html)

<p align="left">
  <a href="https://erdogant.github.io/classeval/pages/html/Examples.html">
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/Figure_1.png" width="600" />
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/Figure_2.png" width="400" />
  </a>
</p>


#

#### [Example: Evaluate multi-class model](https://erdogant.github.io/classeval/pages/html/Examples.html#examples-multi-class-model)

<p align="left">
  <a href="https://erdogant.github.io/classeval/pages/html/Examples.html#examples-multi-class-model">
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/multiclass_fig1_1.png" width="400" />
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/multiclass_fig1_3.png" width="400" />
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/multiclass_fig1_4.png" width="400" />
  </a>
</p>



#### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

#### Contribute
* Contributions are welcome.
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

#### Licence
* See [LICENSE](LICENSE) for details.
