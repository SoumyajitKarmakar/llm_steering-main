"""
neural_controllers_xrfm.py

This module is a drop-in replacement for neural_controllers.py that uses the xRFM 
implementation via control_toolkits_xrfm.py.

Usage:
    # Instead of:
    # from neural_controllers import NeuralController
    
    # Use:
    from neural_controllers_xrfm import NeuralController
"""

import torch
import random
import numpy as np

SEED = 0
random.seed(SEED)               
np.random.seed(SEED)            
torch.manual_seed(SEED)         
torch.cuda.manual_seed(SEED) 


import generation_utils
import direction_utils_xrfm as direction_utils  # Use xRFM version

# Import from xRFM-based control_toolkits
from control_toolkits_xrfm import (
    RFMToolkit,
    LinearProbeToolkit,
    LogisticRegressionToolkit,
    MeanDifferenceToolkit,
    PCAToolkit,
)
from utils import LLMType

import os
import pickle
from tqdm import tqdm
import shutil

TOOLKITS = {
    'rfm' : RFMToolkit,  # This now uses xRFM
    'linear' : LinearProbeToolkit,
    'logistic' : LogisticRegressionToolkit,
    'mean_difference' : MeanDifferenceToolkit,
    'pca' : PCAToolkit
}


# Import the rest of the NeuralController class from the original module
# but override the TOOLKITS mapping
from neural_controllers import NeuralController as _OriginalNeuralController
from neural_controllers import get_non_cross_attention_layer_indices_after_first_cross


class NeuralController(_OriginalNeuralController):
    """
    NeuralController that uses xRFM library for RFM-based steering vector extraction.
    
    This is a drop-in replacement for the original NeuralController.
    The only difference is that the 'rfm' control method uses the xRFM library
    instead of the original adit_rfm.py implementation.
    """
    
    def __init__(self, llm, tokenizer, control_method='rfm', n_components=5, 
                 rfm_iters=8, batch_size=16):
        # Call parent init but override the toolkit
        super().__init__(llm, tokenizer, control_method, n_components, rfm_iters, batch_size)
        
        # Override the toolkit with xRFM version
        self.toolkit = TOOLKITS[control_method]()
        
        if control_method == 'rfm':
            print("Using xRFM library for RFM-based steering vector extraction")
