# -*- coding: utf-8 -*-
"""PyMilo chain for Neighbors models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import SKLEARN_NEIGHBORS_TABLE
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.neighbors_tree_transporter import NeighborsTreeTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

neighbors_chain = AbstractChain(
    {
        "PreprocessingTransporter": PreprocessingTransporter(),
        "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
        "NeighborsTreeTransporter": NeighborsTreeTransporter(),
    },
    SKLEARN_NEIGHBORS_TABLE,
)
