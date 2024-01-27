<div align="center">
    <img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/logo.png" width="500" height="300">
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
PyMilo is an open source Python package that provides a simple, efficient, and safe way for users to export pre-trained machine learning models in a transparent way. By this, the exported model can be used in other environments, transferred across different platforms, and shared with others. PyMilo allows the users to export the models that are trained using popular Python libraries like scikit-learn, and then use them in deployment environments, or share them without exposing the underlying code or dependencies. The transparency of the exported models ensures reliability and safety for the end users, as it eliminates the risks of binary or pickle formats.
</p>
<table>
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
		<td align="center"><a href="https://www.codefactor.io/repository/github/openscilab/pymilo"><img src="https://www.codefactor.io/repository/github/openscilab/pymilo/badge" alt="CodeFactor" /></a></td>
		<td align="center"><a href="https://app.codacy.com/gh/openscilab/pymilo/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/9eeec99ed11f4d9b86af36dc90f5f753"></a></td>
		<td align="center"><a href="https://codebeat.co/projects/github-com-openscilab-pymilo-dev"><img alt="codebeat badge" src="https://codebeat.co/badges/1259254f-39fc-4491-8469-17d8a43b6697" /></a></td>
	</tr>
</table>


## Installation

### PyPI

- Check [Python Packaging User Guide](https://packaging.python.org/installing/)
- Run `pip install pymilo==0.5`
### Source code
- Download [Version 0.5](https://github.com/openscilab/pymilo/archive/v0.5.zip) or [Latest Source](https://github.com/openscilab/pymilo/archive/dev.zip)
- Run `pip install .`

## Usage
### Simple Linear Model Preparation 
```pycon
>>> from sklearn import datasets
>>> from pymilo import Export, Import
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
| Neural networks &#x2705; | -  | 
| Trees &#x2705; | -  | 
| Clustering &#x2705; | -  | 
| Naïve Bayes &#x2705; | -  | 
| Support vector machines (SVMs) &#x274C; | -  | 
| Nearest Neighbors &#x274C; | -  | 
| Ensemble Models &#x274C; | - | 
Details are available in [Supported Models](https://github.com/openscilab/pymilo/blob/main/SUPPORTED_MODELS.md).

## Issues & bug reports

Just fill an issue and describe it. We'll check it ASAP! or send an email to [info@openscilab.com](mailto:info@openscilab.com "info@openscilab.com"). 

- Please complete the issue template
 
You can also join our discord server

<a href="https://discord.gg/mtuMS8AjDS">
  <img src="https://img.shields.io/discord/1064533716615049236.svg?style=for-the-badge" alt="Discord Channel">
</a>


## Show Your Support


### Star this repo

Give a ⭐️ if this project helped you!

### Donate to our project
If you do like our project and we hope that you do, can you please support us? Our project is not and is never going to be working for profit. We need the money just so we can continue doing what we do ;-) .			

<a href="https://openscilab.com/#donation" target="_blank"><img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/donation.png" height="90px" width="270px" alt="PyMilo Donation"></a>
