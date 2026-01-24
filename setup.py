# -*- coding: utf-8 -*-
"""Setup module."""
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

INSTALLATION_MODES = {
    'core': 'requirements.txt',
    'streaming': 'streaming-requirements.txt',
}


def get_requires(mode='core'):
    """Read associated requirements to install."""
    reqs_path = INSTALLATION_MODES[mode]
    requirements = open(reqs_path, "r").read()
    return list(filter(lambda x: x != "", requirements.split()))


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
        return '''PyMilo: Python for ML I/O'''


setup(
    name='pymilo',
    packages=find_packages(include=['pymilo*'], exclude=['tests*']),
    version='1.5',
    description='PyMilo: Python for ML I/O',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    author='PyMilo Development Team',
    author_email='pymilo@openscilab.com',
    url='https://github.com/openscilab/pymilo',
    download_url='https://github.com/openscilab/pymilo/tarball/v1.5',
    keywords="machine_learning ml ai mlops model export import",
    project_urls={
            'Source': 'https://github.com/openscilab/pymilo',
    },
    install_requires=get_requires(),
    extras_require={
        'streaming': get_requires(mode='streaming'),
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
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
    entry_points={
            'console_scripts': [
                'pymilo = pymilo.__main__:main',
            ]
    }
)
