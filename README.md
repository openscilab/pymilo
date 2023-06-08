<div align="center">
    <img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/logo.png" width="550">
    <br/>
    <br/>
    <a href="https://codecov.io/gh/openscilab/pymilo">
        <img src="https://codecov.io/gh/openscilab/pymilo/branch/main/graph/badge.svg" alt="Codecov"/>
    </a>
    <a href="https://badge.fury.io/py/pymilo">
        <img src="https://badge.fury.io/py/pymilo.svg" alt="PyPI version" height="18">
    </a>
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3">
    </a>
    <a href="https://anaconda.org/openscilab/pymilo">
        <img src="https://anaconda.org/openscilab/pymilo/badges/version.svg">
    </a>
    <a href="https://colab.research.google.com/github/openscilab/pymilo/blob/main">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Pymilo-Colab"/>
    </a>
    <a href="https://discord.gg/mtuMS8AjDS">
        <img src="https://img.shields.io/discord/1064533716615049236.svg" alt="Discord Channel">
    </a>
</div>

----------

## Table of contents

* [Overview](https://github.com/openscilab/pymilo#overview)
* [Installation](https://github.com/openscilab/pymilo#installation)
* [Usage](https://github.com/openscilab/pymilo#usage)
* [Issues & Bug Reports](https://github.com/openscilab/pymilo#issues--bug-reports)
* [Todo](https://github.com/openscilab/pymilo/blob/main/TODO.md)
* [Contribution](https://github.com/openscilab/pymilo/blob/main/.github/CONTRIBUTING.md)
* [Authors](https://github.com/openscilab/pymilo/blob/main/AUTHORS.md)
* [License](https://github.com/openscilab/pymilo/blob/main/LICENSE)
* [Show Your Support](https://github.com/openscilab/pymilo#show-your-support)
* [Changelog](https://github.com/openscilab/pymilo/blob/main/CHANGELOG.md)
* [Code of Conduct](https://github.com/openscilab/pymilo/blob/main/.github/CODE_OF_CONDUCT.md)


## Overview
<p align="justify">
Pymilo is an open source Python package that provides a simple, efficient, and safe way for users to export pre-trained machine learning models in a transparent way. By this, the exported model can be used in other environments, transferred across different platforms, and shared with others. Pymilo allows the users to export the models that are trained using popular Python libraries like scikit-learn, and then use them in deployment environments, or share them without exposing the underlying code or dependencies. The transparency of the exported models ensures reliability and safety for the end users, as it eliminates the risks of binary or pickle formats.
</p>
<table>
    <tr>
        <td align="center">Open Hub</td>
        <td align="center">
            <a href="https://www.openhub.net/p/pymilo">
                <img src="https://www.openhub.net/p/pymilo/widgets/project_thin_badge.gif">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">PyPI Counter</td>
        <td align="center">
            <a href="http://pepy.tech/project/pymilo">
                <img src="http://pepy.tech/badge/pymilo">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">Github Stars</td>
        <td align="center">
            <a href="https://github.com/openscilab/pymilo">
                <img src="https://img.shields.io/github/stars/openscilab/pymilo.svg?style=social&label=Stars">
            </a>
        </td>
    </tr>
</table>
<table>
    <tr> 
        <td align="center">Branch</td>
        <td align="center">main</td>
        <td align="center">dev</td>
    </tr>
    <tr>
        <td align="center">CI</td>
        <td align="center">
            <img src="https://github.com/openscilab/pymilo/workflows/CI/badge.svg?branch=main">
        </td>
        <td align="center">
            <img src="https://github.com/openscilab/pymilo/workflows/CI/badge.svg?branch=dev">
            </td>
    </tr>
</table>
<table>
    <tr> 
        <td align="center">Code Quality</td>
        <td align="center">
            <a class="badge-align" href="https://www.codacy.com/app/openscilab/pymilo?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=openscilab/pymilo&amp;utm_campaign=Badge_Grade">
                <img src="https://api.codacy.com/project/badge/Grade/5d9463998a0040d09afc2b80c389365c"/>
            </a>
        </td>
        <td align="center">
            <a href="https://www.codefactor.io/repository/github/openscilab/pymilo/overview/dev">
                <img src="https://www.codefactor.io/repository/github/openscilab/pymilo/badge/dev" alt="CodeFactor"/>
            </a>
        </td>
    </tr>
</table>


## Installation

### Source code
- Download [Version 0.1](https://github.com/openscilab/pymilo/archive/v0.1.zip) or [Latest Source](https://github.com/openscilab/pymilo/archive/dev.zip)
- Run `pip install -r requirements.txt` or `pip3 install -r requirements.txt` (Need root access)
- Run `python3 setup.py install` or `python setup.py install` (Need root access)

### PyPI

- Check [Python Packaging User Guide](https://packaging.python.org/installing/)
- Run `pip install pymilo==0.1` or `pip3 install pymilo==0.1` (Need root access)

### Conda

- Check [Conda Managing Package](https://conda.io/)
- Update Conda using `conda update conda` (Need root access)
- Run `conda install -c openscilab pymilo` (Need root access)

### Easy install

- Run `easy_install --upgrade pymilo` (Need root access)


## Usage
### Simple Linear Model Preparation 
```pycon
>>> from sklearn import datasets
>>> from pymilo.pymilo_obj import Export, Import
>>> from sklearn.linear_model import LinearRegression
>>> import os

>>> X, Y = datasets.load_diabetes(return_X_y=True)
>>> threshold = 20
>>> X_train, X_test = X[:-threshold], X[-threshold:]
>>> Y_train, Y_test = Y[:-threshold], Y[-threshold:]
>>> model = LinearRegression()
>>> #### Train the model using the training sets
>>> model.fit(X_train, Y_train)
```
### Save Model 
```pycon
>>> #### Export the fitted model to a transparent json file
>>> exported_model = Export(model)
>>> PATH_TO_JSON_FILE = os.path.join(os.getcwd(),"test.json")
>>> exported_model.save(PATH_TO_JSON_FILE)
```
### Load Model
```pycon
>>> #### Import the pymilo-exported model and get a real scikit model
>>> imported_model = Import(PATH_TO_JSON_FILE)
```
### Get the associated Scikit model
```pycon 
>>> imported_sklearn_model = imported_model.to_model()
```
#### Note: `imported_sklearn_model` has the **exact same** functionality as the `model` object earlier.

## Supported ML Models
| scikit-learn | PyTorch | 
| ---------------- | ---------------- | 
| Linear Models &#x2705; | - | 
| Neural networks &#x274C; | -  | 
| Clustering &#x274C; | -  | 
| Trees &#x274C; | -  | 
| Ensemble Models &#x274C; | - | 

## Issues & bug reports

1. Fill an issue and describe it. We'll check it ASAP!
    - Please complete the issue template
2. Discord : [https://discord.gg/mtuMS8AjDS](https://discord.gg/mtuMS8AjDS)


## Social Media
1. [Discord](https://discord.gg/6Ce6WuB2)
## Show Your Support


### Star this repo

Give a ⭐️ if this project helped you!

### Donate to our project
If you do like our project and we hope that you do, can you please support us? Our project is not and is never going to be working for profit. We need the money just so we can continue doing what we do ;-) .			

<a href="https://openscilab.com/#donation" target="_blank"><img src="https://github.com/openscilab/static-assets/raw/main/buttons/donation.jpg" height="90px" width="270px" alt="OSL Donation"></a>
