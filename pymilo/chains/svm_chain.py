# -*- coding: utf-8 -*-
"""PyMilo chain for SVM models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import SKLEARN_SVM_TABLE
from ..transporters.preprocessing_transporter import PreprocessingTransporter
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter

svm_chain = AbstractChain(
    {
        "PreprocessingTransporter": PreprocessingTransporter(),
        "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
        "RandomStateTransporter": RandomStateTransporter(),
    },
    SKLEARN_SVM_TABLE,
)
