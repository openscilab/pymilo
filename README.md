<div align="center">
    <img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/logo.png" width="500" height="300">
    <br/>
    <br/>
    <a href="https://codecov.io/gh/openscilab/pymilo"><img src="https://codecov.io/gh/openscilab/pymilo/branch/main/graph/badge.svg" alt="Codecov"/></a>
    <a href="https://badge.fury.io/py/pymilo"><img src="https://badge.fury.io/py/pymilo.svg" alt="PyPI version"></a>
    <a href="https://anaconda.org/openscilab/pymilo"><img src="https://anaconda.org/openscilab/pymilo/badges/version.svg"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3"></a>
    <a href="https://github.com/openscilab/pymilo"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/openscilab/pymilo"></a>
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
	    <a href="https://pepy.tech/projects/pymilo">
	        <img src="https://static.pepy.tech/badge/pymilo" alt="PyPI Downloads">
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
- Run `pip install pymilo==1.2`
### Source code
- Download [Version 1.2](https://github.com/openscilab/pymilo/archive/v1.2.zip) or [Latest Source](https://github.com/openscilab/pymilo/archive/dev.zip)
- Run `pip install .`

### Conda

- Check [Conda Managing Package](https://conda.io/)
- Update Conda using `conda update conda`
- Run `conda install -c openscilab pymilo`


## Usage
### Import/Export
Imagine you want to train a `LinearRegression` model representing this equation: $y = x_0 + 2x_1 + 3$. You will create data points (`X`, `y`) and train your model as follows.
```python
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
# y = 1 * x_0 + 2 * x_1 + 3
model = LinearRegression().fit(X, y)
pred = model.predict(np.array([[3, 5]]))
# pred = [16.] (=1 * 3 + 2 * 5 + 3)
```

Using PyMilo `Export` class you can easily serialize and export your trained model into a JSON file.
```python
from pymilo import Export
Export(model).save("model.json")
```

You can check out your model as a JSON file now.
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
                1.0000000000000002,
                1.9999999999999991
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
                1.618033988749895,
                0.6180339887498948
            ],
            "pymiloed-ndarray-dtype": "float64",
            "pymiloed-ndarray-shape": [
                2
            ],
            "pymiloed-data-structure": "numpy.ndarray"
        },
        "intercept_": {
            "value": 3.0000000000000018,
            "np-type": "numpy.float64"
        }
    },
    "sklearn_version": "1.4.2",
    "pymilo_version": "0.8",
    "model_type": "LinearRegression"
}
```
You can see all the learned parameters of the model in this file and change them if you want. This JSON representation is a transparent version of your model.

Now let's load it back. You can do it easily by using PyMilo `Import` class.
```python
from pymilo import Import
model = Import("model.json").to_model()
pred = model.predict(np.array([[3, 5]]))
# pred = [16.] (=1 * 3 + 2 * 5 + 3)
```
This loaded model is exactly the same as the original trained model.

### ML streaming
You can easily serve your ML model from a remote server using `ML streaming` feature of PyMilo.

⚠️ `ML streaming` feature exists in versions `>=1.0`

⚠️ In order to use `ML streaming` feature, make sure you've installed the `streaming` mode of PyMilo

You can choose either `REST` or `WebSocket` as the communication medium protocol.

#### Server
Let's assume you are in the remote server and you want to import the exported JSON file and start serving your model through `REST` protocol!
```python
from pymilo import Import
from pymilo.streaming import PymiloServer, CommunicationProtocol
my_model = Import("model.json").to_model()
communicator = PymiloServer(
    model=my_model,
    port=8000,
    communication_protocol=CommunicationProtocol["REST"],
    ).communicator
communicator.run()
```
Now `PymiloServer` runs on port `8000` and exposes REST API to `upload`, `download` and retrieve **attributes** either **data attributes** like `model._coef` or **method attributes** like `model.predict(x_test)`.

#### Client
By using `PymiloClient` you can easily connect to the remote `PymiloServer` and execute any functionalities that the given ML model has, let's say you want to run `predict` function on your remote ML model and get the result:
```python
from pymilo.streaming import PymiloClient, CommunicationProtocol
pymilo_client = PymiloClient(
    mode=PymiloClient.Mode.LOCAL,
    server_url="SERVER_URL",
    communication_protocol=CommunicationProtocol["REST"],
    )
pymilo_client.toggle_mode(PymiloClient.Mode.DELEGATE)
result = pymilo_client.predict(x_test)
```

ℹ️ If you've deployed `PymiloServer` locally (on port `8000` for instance), then `SERVER_URL` would be `http://127.0.0.1:8000` or `ws://127.0.0.1:8000` based on the selected protocol for the communication medium.

You can also download the remote ML model into your local and execute functions locally on your model.

Calling `download` function on `PymiloClient` will sync the local model that `PymiloClient` wraps upon with the remote ML model, and it doesn't save model directly to a file.

```python
pymilo_client.download()
```
If you want to save the ML model to a file in your local, you can use `Export` class.
```python
from pymilo import Export
Export(pymilo_client.model).save("model.json")
```
Now that you've synced the remote model with your local model, you can run functions.
```python
pymilo_client.toggle_mode(mode=PymiloClient.Mode.LOCAL)
result = pymilo_client.predict(x_test)
```
`PymiloClient` wraps around the ML model, either to the local ML model or the remote ML model, and you can work with `PymiloClient` in the exact same way that you did with the ML model, you can run exact same functions with same signature.

ℹ️ Through the usage of `toggle_mode` function you can specify whether `PymiloClient` applies requests on the local ML model `pymilo_client.toggle_mode(mode=Mode.LOCAL)` or delegates it to the remote server `pymilo_client.toggle_mode(mode=Mode.DELEGATE)`


## Supported ML models
| scikit-learn | PyTorch | 
| ---------------- | ---------------- | 
| Linear Models &#x2705; | - | 
| Neural Networks &#x2705; | -  | 
| Trees &#x2705; | -  | 
| Clustering &#x2705; | -  | 
| Naïve Bayes &#x2705; | -  | 
| Support Vector Machines (SVMs) &#x2705; | -  | 
| Nearest Neighbors &#x2705; | -  |  
| Ensemble Models &#x2705; | - | 
| Pipeline Model &#x2705; | - |
| Preprocessing Models &#x2705; | - |
| Cross Decomposition Models &#x2705; | - |


Details are available in [Supported Models](https://github.com/openscilab/pymilo/blob/main/SUPPORTED_MODELS.md).

## Issues & bug reports

Just fill an issue and describe it. We'll check it ASAP! or send an email to [pymilo@openscilab.com](mailto:pymilo@openscilab.com "pymilo@openscilab.com"). 

- Please complete the issue template
 
You can also join our discord server

<a href="https://discord.gg/mtuMS8AjDS">
  <img src="https://img.shields.io/discord/1064533716615049236.svg?style=for-the-badge" alt="Discord Channel">
</a>

## Acknowledgments

[Python Software Foundation (PSF)](https://www.python.org/psf/) grants PyMilo library partially for versions **1.0, 1.1**. [PSF](https://www.python.org/psf/) is the organization behind Python. Their mission is to promote, protect, and advance the Python programming language and to support and facilitate the growth of a diverse and international community of Python programmers.

<a href="https://www.python.org/psf/"><img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/psf.png" height="65px" alt="Python Software Foundation"></a>

[Trelis Research](https://trelis.com/) grants PyMilo library partially for version **1.0**. [Trelis Research](https://trelis.com/) provides tools and tutorials for businesses and developers looking to fine-tune and deploy large language models.

<a href="https://trelis.com/"><img src="https://trelis.com/wp-content/uploads/2023/10/android-chrome-512x512-1.png" height="75px" alt="Trelis Research"></a>

## Cite

If you use PyMilo in your research, we would appreciate citations to the following paper :

[Rostami, A., Haghighi, S., Sabouri, S., & Zolanvari, A. (2024). *PyMilo: A Python Library for ML I/O*. *arXiv e-prints*, arXiv-2501.](https://arxiv.org/abs/2501.00528)

```bibtex
@article{rostami2024pymilo,
  title={PyMilo: A Python Library for ML I/O},
  author={Rostami, AmirHosein and Haghighi, Sepand and Sabouri, Sadra and Zolanvari, Alireza},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2024}
}
```

## Show your support


### Star this repo

Give a ⭐️ if this project helped you!

### Donate to our project
If you do like our project and we hope that you do, can you please support us? Our project is not and is never going to be working for profit. We need the money just so we can continue doing what we do ;-) .			

<a href="https://openscilab.com/#donation" target="_blank"><img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/donation.png" height="90px" width="270px" alt="PyMilo Donation"></a>
