# -*- coding: utf-8 -*-
"""PyMilo chain for Cross Decomposition models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import SKLEARN_CROSS_DECOMPOSITION_TABLE
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

cross_decomposition_chain = AbstractChain(
    {
        "PreprocessingTransporter": PreprocessingTransporter(),
        "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    },
    SKLEARN_CROSS_DECOMPOSITION_TABLE,
)
