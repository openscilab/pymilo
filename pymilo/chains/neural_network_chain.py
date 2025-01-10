# -*- coding: utf-8 -*-
"""PyMilo chain for Neural Network models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import SKLEARN_NEURAL_NETWORK_TABLE
from ..transporters.adamoptimizer_transporter import AdamOptimizerTransporter
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.sgdoptimizer_transporter import SGDOptimizerTransporter

neural_network_chain = AbstractChain(
    {
        "PreprocessingTransporter": PreprocessingTransporter(),
        "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
        "RandomStateTransporter": RandomStateTransporter(),
        "SGDOptimizer": SGDOptimizerTransporter(),
        "AdamOptimizerTransporter": AdamOptimizerTransporter(),
    },
    SKLEARN_NEURAL_NETWORK_TABLE,
)
