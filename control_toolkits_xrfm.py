"""
control_toolkits_xrfm.py

This module is a drop-in replacement for control_toolkits.py that uses the xRFM 
implementation via direction_utils_xrfm.py.

Only the RFMToolkit class is modified to use direction_utils_xrfm.
All other toolkit classes are imported from the original module.
"""

import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
import direction_utils_xrfm as direction_utils  # Use xRFM version
from utils import split_indices
from sklearn.metrics import log_loss


from tqdm import tqdm
import time
from copy import deepcopy

# Import other toolkits from original module (they don't use RFM)
from control_toolkits import (
    LinearProbeToolkit,
    LogisticRegressionToolkit,
    MeanDifferenceToolkit,
    PCAToolkit,
)


class RFMToolkit():
    """RFM Toolkit using xRFM library instead of adit_rfm."""
    
    def __init__(self):
        pass

    def _compute_directions(self, data, labels, llm, model, tokenizer, hidden_layers, hyperparams,
                            test_data=None, test_labels=None, **kwargs):
        
        top_eigs = kwargs.get('top_eigs', 25) 
        compare_to_linear = kwargs.get('compare_to_linear', False)
        log_spectrum = kwargs.get('log_spectrum', False)
        log_path = kwargs.get('log_path', None)
                
        train_indices, val_indices = split_indices(len(data))
        test_data_provided = test_data is not None 
        
        all_y = labels.float().cuda()
        train_y = all_y[train_indices]
        val_y = all_y[val_indices]
        num_classes = all_y.shape[1]
        
        direction_outputs = {
                                'val' : [],
                                'test' : []
                            }
        
        predictor_outputs = {
                                'val' : [],
                                'test' : []
                            }

        hidden_states = direction_utils.get_hidden_states(data, llm, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])
        if test_data_provided:
            test_hidden_states = direction_utils.get_hidden_states(test_data, llm, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])
            test_direction_accs = {}
            test_predictor_accs = {}            
            test_y = torch.tensor(test_labels).reshape(-1,1).float().cuda()
                        
        n_components = hyperparams['n_components']
        directions = {}
        detector_coefs = {}

        for layer_to_eval in tqdm(hidden_layers):
            hidden_states_at_layer = hidden_states[layer_to_eval].cuda().float()
            train_X = hidden_states_at_layer[train_indices] 
            val_X = hidden_states_at_layer[val_indices]
                
            assert(len(train_X) == len(train_y))
            assert(len(val_X) == len(val_y))

            # Use the xRFM-based train_rfm_probe_on_concept
            u = direction_utils.train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, hyperparams)
            
            if u is None:
                directions[layer_to_eval] = torch.zeros(1, train_X.shape[1], device='cuda')
            else:
                directions[layer_to_eval] = u.reshape(1, -1)
            
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(hidden_states, all_y, directions, n_components)
            for layer_to_eval in tqdm(hidden_layers):
                for c_idx in range(n_components):
                    directions[layer_to_eval][c_idx] *= signs[layer_to_eval][c_idx]
                
        return directions, signs, detector_coefs, None, None

    def _compute_signs(self, hidden_states, all_y, directions, n_components):
        
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            for c_idx in range(n_components):
                direction = directions[layer][c_idx]
                hidden_state_projections = direction_utils.project_onto_direction(xs, direction).to(all_y.device)
                sign = 2*(direction_utils.pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
                signs[layer][c_idx] = sign.item()

        return signs
