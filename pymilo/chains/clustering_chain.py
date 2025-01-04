# -*- coding: utf-8 -*-
"""PyMilo chain for Clustering models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import SKLEARN_CLUSTERING_TABLE, NOT_SUPPORTED
from ..transporters.cfnode_transporter import CFNodeTransporter
from ..transporters.function_transporter import FunctionTransporter
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

CLUSTERING_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "FunctionTransporter": FunctionTransporter(),
    "CFNodeTransporter": CFNodeTransporter(),
}

if SKLEARN_CLUSTERING_TABLE["BisectingKMeans"] != NOT_SUPPORTED:
    from ..transporters.bisecting_tree_transporter import BisectingTreeTransporter
    from ..transporters.randomstate_transporter import RandomStateTransporter
    CLUSTERING_CHAIN["RandomStateTransporter"] = RandomStateTransporter()
    CLUSTERING_CHAIN["BisectingTreeTransporter"] = BisectingTreeTransporter()

clustering_chain = AbstractChain(CLUSTERING_CHAIN, SKLEARN_CLUSTERING_TABLE)
