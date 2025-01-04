# -*- coding: utf-8 -*-
"""PyMilo chain for Naive Bayes models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import SKLEARN_NAIVE_BAYES_TABLE
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

naive_bayes_chain = AbstractChain(
    {
        "PreprocessingTransporter": PreprocessingTransporter(),
        "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    },
    SKLEARN_NAIVE_BAYES_TABLE,
)
