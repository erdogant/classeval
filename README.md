# classeval

[![Python](https://img.shields.io/pypi/pyversions/classeval)](https://img.shields.io/pypi/pyversions/classeval)
[![PyPI Version](https://img.shields.io/pypi/v/classeval)](https://pypi.org/project/classeval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/classeval/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/classeval/week)](https://pepy.tech/project/classeval/week)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* classeval is Python package

### Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install classeval from PyPI (recommended). classeval is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

### Requirements
* It is advisable to create a new environment. 
```python
conda create -n env_classeval python=3.6
conda activate env_classeval
pip install -r requirements
```

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
import classeval as classeval
```

#### Example:
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/classeval/data/example_data.csv')
model = classeval.fit(df)
G = classeval.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/Figure_1.png" width="600" />
  <img src="https://github.com/erdogant/classeval/blob/master/docs/figs/Figure_2.png" width="400" />
</p>


#### Citation
Please cite classeval in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2020classeval,
  title={classeval},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/classeval}},
}
```

#### References
* 
   
#### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

#### Contribute
* Contributions are welcome.

#### Licence
See [LICENSE](LICENSE) for details.

#### Donation
* This work is created and maintained in my free time. Contributions of any kind are very appreciated. <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Sponsering</a> is also possible.

