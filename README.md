<div align="center">
    <img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/logo.png" width="500" height="300">
    <br/>
    <br/>
    <a href="https://codecov.io/gh/openscilab/pymilo"><img src="https://codecov.io/gh/openscilab/pymilo/branch/main/graph/badge.svg" alt="Codecov"/></a>
    <a href="https://badge.fury.io/py/pymilo"><img src="https://badge.fury.io/py/pymilo.svg" alt="PyPI version" height="18"></a>
    <a href="https://anaconda.org/openscilab/pymilo"><img src="https://anaconda.org/openscilab/pymilo/badges/version.svg"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3"></a>
    <a href="https://discord.gg/mtuMS8AjDS"><img src="https://img.shields.io/discord/1064533716615049236.svg" alt="Discord Channel"></a>
</div>

----------

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
            <img src="https://github.com/openscilab/pymilo/actions/workflows/test.yml/badge.svg?branch=main">
        </td>
        <td align="center">
            <img src="https://github.com/openscilab/pymilo/actions/workflows/test.yml/badge.svg?branch=dev">
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
- Run `pip install pymilo==0.8`
### Source code
- Download [Version 0.8](https://github.com/openscilab/pymilo/archive/v0.8.zip) or [Latest Source](https://github.com/openscilab/pymilo/archive/dev.zip)
- Run `pip install .`

### Conda

- Check [Conda Managing Package](https://conda.io/)
- Update Conda using `conda update conda`
- Run `conda install -c openscilab pymilo`


## Usage
Let's say you want to train your `LinearRegression` model by your data `X`, `y`. After splitting the data into train/test segments you would have `X_train` and `y_train` as your training data. So you train your model.
```pycon
>>> model = LinearRegression()
>>> model.fit(X_train, Y_train)
```

### Save model - Serialize
Now, let's say you want to have a transparent `.json` file from your trained model. Using PyMilo you can easily have than.
```pycon
>>> from pymilo import Export
>>> exported_model = Export(model)
>>> exported_model.save("model.json")
```
You can now see your model as a json file. It should be something like bellow.
```json
{
    "data": {
        "fit_intercept": true,
        "copy_X": true,
        "n_jobs": null,
        "positive": false,
        "n_features_in_": 2,
        "coef_": {
            "pymiloed-ndarray-list": [
                0.3060942475420746,
                -237.63557011300682,
            ],
            "pymiloed-ndarray-dtype": "float64",
            "pymiloed-ndarray-shape": [
                2
            ],
            "pymiloed-data-structure": "numpy.ndarray"
        },
        "rank_": 2,
        "singular_": {
            "pymiloed-ndarray-list": [
                1.9578051002417807,
                1.1797491126040702,
            ],
            "pymiloed-ndarray-dtype": "float64",
            "pymiloed-ndarray-shape": [
                2
            ],
            "pymiloed-data-structure": "numpy.ndarray"
        },
        "intercept_": {
            "value": 152.76429169049118,
            "np-type": "numpy.float64"
        }
    },
    "sklearn_version": "1.2.2",
    "pymilo_version": "0.8",
    "model_type": "LinearRegression"
}
```
In this file you can see all the learned parameter from the model. You can even change them as you want. This json representation is a transparent and safe version of your model.

### Load model
Now let's load it back. You can do it easily by using PyMilo `Export`.
```pycon
>>> from pymilo import Export
>>> imported_model = Import("model.json")
>>> model = imported_model.to_model()
```
This model is exactly as same as the firstly trained model.

## Supported ML models
| scikit-learn | PyTorch | 
| ---------------- | ---------------- | 
| Linear Models &#x2705; | - | 
| Neural networks &#x2705; | -  | 
| Trees &#x2705; | -  | 
| Clustering &#x2705; | -  | 
| Naïve Bayes &#x2705; | -  | 
| Support vector machines (SVMs) &#x2705; | -  | 
| Nearest Neighbors &#x2705; | -  |  
| Ensemble Models &#x2705; | - | 
| Pipeline Model &#x2705; | - |
| Preprocessing Models &#x2705; | - |

Details are available in [Supported Models](https://github.com/openscilab/pymilo/blob/main/SUPPORTED_MODELS.md).

## Issues & bug reports

Just fill an issue and describe it. We'll check it ASAP! or send an email to [pymilo@openscilab.com](mailto:pymilo@openscilab.com "pymilo@openscilab.com"). 

- Please complete the issue template
 
You can also join our discord server

<a href="https://discord.gg/mtuMS8AjDS">
  <img src="https://img.shields.io/discord/1064533716615049236.svg?style=for-the-badge" alt="Discord Channel">
</a>


## Show your support


### Star this repo

Give a ⭐️ if this project helped you!

### Donate to our project
If you do like our project and we hope that you do, can you please support us? Our project is not and is never going to be working for profit. We need the money just so we can continue doing what we do ;-) .			

<a href="https://openscilab.com/#donation" target="_blank"><img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/donation.png" height="90px" width="270px" alt="PyMilo Donation"></a>