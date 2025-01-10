# -*- coding: utf-8 -*-
"""PyMilo chain for Decision Trees models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import SKLEARN_DECISION_TREE_TABLE
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.tree_transporter import TreeTransporter

decision_trees_chain = AbstractChain(
    {
        "PreprocessingTransporter": PreprocessingTransporter(),
        "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
        "RandomStateTransporter": RandomStateTransporter(),
        "TreeTransporter": TreeTransporter(),
    },
    SKLEARN_DECISION_TREE_TABLE,
)
