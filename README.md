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
	</tr>
</table>


## Installation

### PyPI

- Check [Python Packaging User Guide](https://packaging.python.org/installing/)
- Run `pip install pymilo==1.5`
### Source code
- Download [Version 1.5](https://github.com/openscilab/pymilo/archive/v1.5.zip) or [Latest Source](https://github.com/openscilab/pymilo/archive/dev.zip)
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

#### Export

The `Export` class facilitates exporting of machine learning models to JSON files.

| **Parameter** | **Description** |
| ------------- | --------------- |
| model | The machine learning model to be exported |

| **Property** | **Description** |
| ------------ | --------------- |
| data | The serialized model data including all learned parameters |
| version | The scikit-learn version used to train the model |
| type | The type/class name of the exported model |

| **Method** | **Description** |
| ---------- | --------------- |
| save | Save the exported model to a JSON file |
| to_json | Return the model as a JSON string representation |
| batch_export | Export multiple models to individual JSON files in a directory |

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

#### Import

The `Import` class facilitates importing of serialized models from JSON files, JSON strings, or URLs.

| **Parameter** | **Description** |
| ------------- | --------------- |
| file_adr | Path to the JSON file containing the serialized model |
| json_dump | JSON string representation of the serialized model |
| url | URL to download the serialized model from |

| **Property** | **Description** |
| ------------ | --------------- |
| data | The deserialized model data |
| version | The scikit-learn version of the original model |
| type | The type/class name of the imported model |

| **Method** | **Description** |
| ---------- | --------------- |
| to_model | Convert the imported data back to a scikit-learn model |
| batch_import | Import multiple models from JSON files in a directory |

This loaded model is exactly the same as the original trained model.

### ML streaming
You can easily serve your ML model from a remote server using `ML streaming` feature of PyMilo.

⚠️ `ML streaming` feature exists in versions `>=1.0`

⚠️ In order to use `ML streaming` feature, make sure you've installed the `streaming` mode of PyMilo

⚠️ The `ML streaming` feature is under construction and is not yet considered stable.

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

#### PymiloServer

The `PymiloServer` class facilitates streaming machine learning models over a network.

| **Parameter** | **Description** |
| ------------- | --------------- |
| port | Port number for the server to listen on (default: 8000) |
| host | Host address for the server (default: "127.0.0.1") |
| compressor | Compression method from `Compression` enum |
| communication_protocol | Communication protocol from `CommunicationProtocol` enum |

The `compressor` parameter accepts values from the `Compression` enum including `NULL` (no compression), `GZIP`, `ZLIB`, `LZMA`, or `BZ2`. The `communication_protocol` parameter accepts values from the `CommunicationProtocol` enum including `REST` or `WEBSOCKET`.

| **Method** | **Description** |
| ---------- | --------------- |
| init_client | Initialize a new client with the given client ID |
| remove_client | Remove an existing client by client ID |
| init_ml_model | Initialize a new ML model for a given client |
| set_ml_model | Set or update the ML model for a client |
| remove_ml_model | Remove an existing ML model for a client |
| get_ml_models | Get all ML model IDs for a client |
| execute_model | Execute model methods or access attributes |
| grant_access | Allow a client to access another client's model |
| revoke_access | Revoke access to a client's model |
| get_allowed_models | Get models a client is allowed to access |

Now `PymiloServer` runs on port `8000` and exposes REST API to `upload`, `download` and retrieve **attributes** either **data attributes** like `model._coef` or **method attributes** like `model.predict(x_test)`.

ℹ️ By default, `PymiloServer` listens on the loopback interface (`127.0.0.1`). To make it accessible over a local network (LAN), specify your machine’s LAN IP address in the `host` parameter of the `PymiloServer` constructor.

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

#### PymiloClient

The `PymiloClient` class facilitates working with remote PyMilo servers.

| **Parameter** | **Description** |
| ------------- | --------------- |
| model | The local ML model to wrap around |
| mode | Operating mode (LOCAL or DELEGATE) |
| compressor | Compression method from `Compression` enum |
| server_url | URL of the PyMilo server |
| communication_protocol | Communication protocol from `CommunicationProtocol` enum |

The `mode` parameter accepts two values `LOCAL` to execute operations on the local model, or `DELEGATE` to delegate operations to the remote server. The `compressor` parameter accepts values from the `Compression` enum including `NULL` (no compression), `GZIP`, `ZLIB`, `LZMA`, or `BZ2`. The `communication_protocol` parameter accepts values from the `CommunicationProtocol` enum including `REST` or `WEBSOCKET`.

| **Method** | **Description** |
| ---------- | --------------- |
| toggle_mode | Switch between LOCAL and DELEGATE modes |
| register | Register the client with the remote server |
| deregister | Deregister the client from the server |
| register_ml_model | Register an ML model with the server |
| deregister_ml_model | Deregister an ML model from the server |
| upload | Upload the local model to the remote server |
| download | Download the remote model to local |
| get_ml_models | Get all registered ML models for this client |
| grant_access | Grant access to this client's model to another client |
| revoke_access | Revoke access previously granted to another client |
| get_allowance | Get clients who have access to this client's models |
| get_allowed_models | Get models this client is allowed to access from another client |

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
| Feature Extractor Models &#x2705; | - |
| Composite Models &#x2705; | - |


Details are available in [Supported Models](https://github.com/openscilab/pymilo/blob/main/SUPPORTED_MODELS.md).

## Issues & bug reports

Just fill an issue and describe it. We'll check it ASAP! or send an email to [pymilo@openscilab.com](mailto:pymilo@openscilab.com "pymilo@openscilab.com"). 

- Please complete the issue template
 
You can also join our discord server

<a href="https://discord.gg/mtuMS8AjDS">
  <img src="https://img.shields.io/discord/1064533716615049236.svg?style=for-the-badge" alt="Discord Channel">
</a>

## Contributing

We welcome contributions! Please read our **[Contributing Guidelines](.github/CONTRIBUTING.md)** before submitting any changes.

## Acknowledgments

[Python Software Foundation (PSF)](https://www.python.org/psf/) grants PyMilo library partially for versions **1.0, 1.1**. [PSF](https://www.python.org/psf/) is the organization behind Python. Their mission is to promote, protect, and advance the Python programming language and to support and facilitate the growth of a diverse and international community of Python programmers.

<a href="https://www.python.org/psf/"><img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/psf.png" height="65px" alt="Python Software Foundation"></a>

[Trelis Research](https://trelis.com/) grants PyMilo library partially for version **1.0**. [Trelis Research](https://trelis.com/) provides tools and tutorials for businesses and developers looking to fine-tune and deploy large language models.

<a href="https://trelis.com/"><img src="https://trelis.com/wp-content/uploads/2023/10/android-chrome-512x512-1.png" height="75px" alt="Trelis Research"></a>

## Cite

If you use PyMilo in your research, we would appreciate citations to the following paper:

[Rostami, A., Haghighi, S., Sabouri, S. and Zolanvari, A., 2025. PyMilo: A Python Library for ML I/O. *Journal of Open Source Software*, 10(116), p.8858.](https://joss.theoj.org/papers/10.21105/joss.08858)

```bibtex
@article{Rostami2025,
  doi = {10.21105/joss.08858},
  url = {https://doi.org/10.21105/joss.08858},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {116},
  pages = {8858},
  author = {Rostami, AmirHosein and Haghighi, Sepand and Sabouri, Sadra and Zolanvari, Alireza},
  title = {PyMilo: A Python Library for ML I/O},
  journal = {Journal of Open Source Software}
}
```

Download [PyMilo.bib](https://raw.githubusercontent.com/openscilab/pymilo/main/paper/PyMilo.bib)

<a href="https://doi.org/10.21105/joss.08858">
  <img src="https://joss.theoj.org/papers/10.21105/joss.08858/status.svg" alt="JOSS DOI: 10.21105/joss.08858">
</a>

## Show your support


### Star this repo

Give a ⭐️ if this project helped you!

### Donate to our project
If you do like our project and we hope that you do, can you please support us? Our project is not and is never going to be working for profit. We need the money just so we can continue doing what we do ;-) .			

<a href="https://openscilab.com/#donation" target="_blank"><img src="https://github.com/openscilab/pymilo/raw/main/otherfiles/donation.png" height="90px" width="270px" alt="PyMilo Donation"></a>
