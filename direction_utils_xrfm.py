"""
direction_utils_xrfm.py

This module is a drop-in replacement for direction_utils.py that uses the xRFM 
implementation from different_implementation/xRFM-main instead of adit_rfm.py.

Only the train_rfm_probe_on_concept function is modified. All other functions
are imported directly from the original direction_utils module.
"""

import sys
import os

# Add the xRFM package to the path
XRFM_PATH = os.path.join(os.path.dirname(__file__), 'different_implementation', 'xRFM-main')
if XRFM_PATH not in sys.path:
    sys.path.insert(0, XRFM_PATH)

import torch
import numpy as np
from copy import deepcopy

# Import everything from the original direction_utils
from direction_utils import (
    batch_transpose_multiply,
    accuracy_fn,
    pearson_corr,
    split_data,
    precision_score,
    recall_score,
    f1_score,
    compute_classification_metrics,
    get_hidden_states,
    project_hidden_states,
    aggregate_projections_on_coefs,
    project_onto_direction,
    fit_pca_model,
    append_one,
    linear_solve,
    logistic_solve,
    aggregate_layers,
    train_linear_probe_on_concept,
)

# Import the xRFM library
from xrfm.rfm_src import RFM


def train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, hyperparams,
                               bws=[1, 10, 100],
                               regs=[1e-3]):
    """
    Train an RFM probe to find the steering vector for a concept.
    
    This version uses the xRFM library instead of adit_rfm.py.
    
    The original adit_rfm.py version:
    - Loops over bandwidths (bws) and norm=[True, False]
    - Uses fixed reg=1e-3
    - Calls adit_rfm.rfm() which internally:
      - Centers the data (subtracts mean)
      - Optionally normalizes each sample to unit norm (if norm=True)
      - Does iterative kernel ridge regression + AGOP update
      - Returns top eigenvector of final AGOP matrix
    
    This xRFM version mimics the same behavior:
    - Loops over bandwidths (bws) and norm=[True, False]
    - Uses fixed reg=1e-3
    - xRFM internally does the same algorithm
    
    Args:
        train_X: Training hidden states (n_train, d)
        train_y: Training labels (n_train, 1) 
        val_X: Validation hidden states (n_val, d)
        val_y: Validation labels (n_val, 1)
        hyperparams: Dictionary of hyperparameters (unused in this version, 
                     kept for API compatibility)
        bws: List of bandwidths to try
        regs: List of regularization values to try (unused, kept for API compatibility)
        
    Returns:
        best_u: The top eigenvector of the learned AGOP matrix (steering vector)
    """
    
    best_u = None
    best_metric = -float('inf')
    best_bw = None
    best_norm = None
    
    # Fixed reg like original
    reg = 1e-3
    
    # Ensure data is on GPU and in correct format
    if not train_X.is_cuda:
        train_X = train_X.cuda()
    if not train_y.is_cuda:
        train_y = train_y.cuda()
    if not val_X.is_cuda:
        val_X = val_X.cuda()
    if not val_y.is_cuda:
        val_y = val_y.cuda()
        
    # Ensure y has correct shape (n, 1)
    if len(train_y.shape) == 1:
        train_y = train_y.unsqueeze(1)
    if len(val_y.shape) == 1:
        val_y = val_y.unsqueeze(1)
    
    # Match original: loop over bws and norm (True/False)
    for bw in bws:
        for norm in [True, False]:
            try:
                # Preprocess data like adit_rfm.rfm does
                # 1. Center the data
                mean = torch.mean(train_X, dim=0, keepdim=True)
                train_X_processed = train_X - mean
                val_X_processed = val_X - mean
                
                # 2. Optionally normalize to unit norm
                if norm:
                    train_X_processed = train_X_processed / (torch.norm(train_X_processed, dim=-1, keepdim=True) + 1e-8)
                    val_X_processed = val_X_processed / (torch.norm(val_X_processed, dim=-1, keepdim=True) + 1e-8)
                
                # Create RFM model with xRFM library
                model = RFM(
                    kernel='laplace',  # Use standard Laplace kernel (L2 distance)
                    bandwidth=bw,
                    iters=10,  # Same as num_iters in adit_rfm
                    device='cuda',
                    verbose=False,
                    tuning_metric='mse',  # Use MSE for regression-style steering
                )
                
                # Fit the model
                model.fit(
                    train_data=(train_X_processed, train_y),
                    val_data=(val_X_processed, val_y),
                    reg=reg,
                    method='lstsq',  # Use least squares solver
                    verbose=False,
                    early_stop_rfm=True,
                )
                
                # Get the learned AGOP matrix M
                M = model.M
                
                if M is None:
                    continue
                    
                # Extract the top eigenvector (steering direction)
                try:
                    # Use lobpcg for efficiency on large matrices
                    eigenvalues, eigenvectors = torch.lobpcg(M.float(), k=1)
                    u = eigenvectors[:, 0]  # Top eigenvector
                except:
                    # Fallback to full eigendecomposition if lobpcg fails
                    eigenvalues, eigenvectors = torch.linalg.eigh(M.float())
                    u = eigenvectors[:, -1]  # Last eigenvector (largest eigenvalue)
                
                # Compute correlation with validation labels as metric (same as adit_rfm)
                preds = val_X_processed @ u.unsqueeze(1)
                val_r = torch.abs(torch.corrcoef(
                    torch.cat((preds, val_y.float()), dim=-1).T
                ))[0, 1].item()
                
                if val_r >= best_metric:  # >= to match original behavior
                    best_metric = val_r
                    best_u = u.clone()
                    best_bw = bw
                    best_norm = norm
                    
            except Exception as e:
                print(f"xRFM failed with bw={bw}, norm={norm}: {e}")
                continue
                
            torch.cuda.empty_cache()
    
    print(f'Best xRFM r: {best_metric}, reg: {reg}, bw: {best_bw}, norm: {best_norm}')
    
    return best_u


# =============================================================================
# TEST BLOCK - Run this to compare with original direction_utils.py
# =============================================================================

if __name__ == "__main__":
    """
    Test script to verify that train_rfm_probe_on_concept produces similar results
    to the original adit_rfm implementation.
    
    Usage:
        # Test xRFM version:
        python direction_utils_xrfm.py
        
        # To test original version, run this separately:
        # (Copy the test code below and run it with direction_utils.py)
    """
    import time
    
    print("="*70)
    print("Testing train_rfm_probe_on_concept (xRFM version)")
    print("="*70)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic test data
    # Simulate hidden states from an LLM (e.g., 4096 dimensional)
    n_train = 200
    n_val = 50
    d = 512  # Use smaller dimension for faster testing
    
    # Create data with a known structure:
    # The "concept" is encoded in the first few dimensions
    concept_dims = 5
    
    # Generate random hidden states
    train_X = torch.randn(n_train, d).cuda()
    val_X = torch.randn(n_val, d).cuda()
    
    # Create labels based on the first concept dimension (binary)
    train_y = (train_X[:, 0] > 0).float().unsqueeze(1).cuda()
    val_y = (val_X[:, 0] > 0).float().unsqueeze(1).cuda()
    
    # Inject some signal into the data to make the concept learnable
    # Positive examples get a boost in the first few dimensions
    train_X[:, :concept_dims] += train_y * 2.0
    val_X[:, :concept_dims] += val_y * 2.0
    
    print(f"Train X shape: {train_X.shape}")
    print(f"Train y shape: {train_y.shape}")
    print(f"Val X shape: {val_X.shape}")
    print(f"Val y shape: {val_y.shape}")
    print(f"Train y distribution: {train_y.sum().item()}/{len(train_y)} positive")
    print()
    
    # Test hyperparams (not used by our implementation but required for API)
    hyperparams = {'n_components': 1, 'rfm_iters': 10, 'forward_batch_size': 16}
    
    # Run the xRFM version
    print("Running xRFM version...")
    start_time = time.time()
    u_xrfm = train_rfm_probe_on_concept(
        train_X.clone(), train_y.clone(), 
        val_X.clone(), val_y.clone(), 
        hyperparams,
        bws=[1, 10, 100]
    )
    xrfm_time = time.time() - start_time
    
    print(f"\nxRFM completed in {xrfm_time:.2f}s")
    
    if u_xrfm is not None:
        # Flatten to 1D if needed
        u_xrfm = u_xrfm.flatten()
        
        print(f"Steering vector shape: {u_xrfm.shape}")
        print(f"Steering vector norm: {torch.norm(u_xrfm).item():.4f}")
        
        # Check if the steering vector captures the concept
        # The first few dimensions should have higher weights
        top_k = 10
        top_indices = torch.argsort(torch.abs(u_xrfm), descending=True)[:top_k]
        print(f"\nTop {top_k} dimensions (by absolute weight):")
        for i, idx in enumerate(top_indices):
            print(f"  Dim {idx.item()}: {u_xrfm[idx].item():.4f}")
        
        # Compute validation accuracy using the steering vector
        # Reshape to column vector for matrix multiplication
        val_preds = val_X @ u_xrfm.reshape(-1, 1)
        val_corr = torch.corrcoef(torch.cat((val_preds, val_y), dim=-1).T)[0, 1].item()
        print(f"\nValidation correlation: {val_corr:.4f}")
    else:
        print("ERROR: Steering vector is None!")
    
    print()
    print("="*70)
    print("To compare with original, run this in a separate Python session:")
    print("="*70)
    print("""
# Copy this code to test the original direction_utils.py:

import torch
import numpy as np
import time

# Set seeds for reproducibility (same as above)
torch.manual_seed(42)
np.random.seed(42)

# Create SAME synthetic test data
n_train = 200
n_val = 50
d = 512
concept_dims = 5

train_X = torch.randn(n_train, d).cuda()
val_X = torch.randn(n_val, d).cuda()

train_y = (train_X[:, 0] > 0).float().unsqueeze(1).cuda()
val_y = (val_X[:, 0] > 0).float().unsqueeze(1).cuda()

train_X[:, :concept_dims] += train_y * 2.0
val_X[:, :concept_dims] += val_y * 2.0

hyperparams = {'n_components': 1, 'rfm_iters': 10, 'forward_batch_size': 16}

# Import and run original
from direction_utils import train_rfm_probe_on_concept

print("Running original adit_rfm version...")
start_time = time.time()
u_orig = train_rfm_probe_on_concept(
    train_X.clone(), train_y.clone(), 
    val_X.clone(), val_y.clone(), 
    hyperparams,
    bws=[1, 10, 100]
)
orig_time = time.time() - start_time

print(f"Original completed in {orig_time:.2f}s")
if u_orig is not None:
    # Flatten to 1D if needed (handles both 1D and 2D outputs)
    u_orig = u_orig.flatten()
    
    print(f"Steering vector shape: {u_orig.shape}")
    print(f"Steering vector norm: {torch.norm(u_orig).item():.4f}")
    
    top_k = 10
    top_indices = torch.argsort(torch.abs(u_orig), descending=True)[:top_k]
    print(f"Top {top_k} dimensions:")
    for i, idx in enumerate(top_indices):
        print(f"  Dim {idx.item()}: {u_orig[idx].item():.4f}")
    
    # Reshape to column vector for matrix multiplication
    val_preds = val_X @ u_orig.reshape(-1, 1)
    val_corr = torch.corrcoef(torch.cat((val_preds, val_y), dim=-1).T)[0, 1].item()
    print(f"Validation correlation: {val_corr:.4f}")
""")
