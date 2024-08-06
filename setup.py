# -*- coding: utf-8 -*-
"""Setup module."""
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


STREAMING_REQUIREMENTS = [
    "uvicorn",
    "fastapi",
    "requests"
]


def get_extra_ml_streaming_requires():
    """Read requirements.txt."""
    requirements = open("requirements.txt", "r").read()
    all_reqs = list(filter(lambda x: x != "", requirements.split()))
    return [req for req in all_reqs if any(streaming in req for streaming in STREAMING_REQUIREMENTS)]


def get_core_requires():
    """Read requirements.txt."""
    requirements = open("requirements.txt", "r").read()
    all_reqs = list(filter(lambda x: x != "", requirements.split()))
    return [req for req in all_reqs if not any(streaming in req for streaming in STREAMING_REQUIREMENTS)]


def read_description():
    """Read README.md and CHANGELOG.md."""
    try:
        with open("README.md") as r:
            description = "\n"
            description += r.read()
        with open("CHANGELOG.md") as c:
            description += "\n"
            description += c.read()
        return description
    except Exception:
        return '''Transportation of ML models'''


setup(
    name='pymilo',
    packages=find_packages(include=['pymilo*'], exclude=['tests*']),
    version='0.9',
    description='Transportation of ML models',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    author='PyMilo Development Team',
    author_email='pymilo@openscilab.com',
    url='https://github.com/openscilab/pymilo',
    download_url='https://github.com/openscilab/pymilo/tarball/v0.9',
    keywords="python3 python machine_learning ML",
    project_urls={
            'Source': 'https://github.com/openscilab/pymilo',
    },
    install_requires=get_core_requires(),
    extras_require={
        'streaming': get_extra_ml_streaming_requires(),
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    license='MIT',
)
