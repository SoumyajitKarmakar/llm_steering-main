import sys
import os
from pathlib import Path

notebook_path = "/u/skarmakar1/version_check/llm_steering-main/sk"
sys.path.append(notebook_path)

import torch
import numpy as np

from inversion_utils import *
import pickle
from sklearn.model_selection import train_test_split

SEED = 0

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True

#  python runner2.py > RR_logs/log_force1.txt 2>&1


# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# # thresh = 0.01
# thresh = 0.1
# test_weights, test_biases = force_ones(lrr_models, thresh=thresh)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_({thresh}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)


# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# # thresh = 0.005
# thresh = 0.5
# test_weights, test_biases = force_ones(lrr_models, thresh=thresh)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_({thresh}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)

# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models_04.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# thresh = 0.5
# test_weights, test_biases = force_ones(lrr_models, thresh=thresh)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_04({thresh}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)


# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models_07.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# thresh = 0.5
# test_weights, test_biases = force_ones(lrr_models, thresh=thresh)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_07({thresh}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)

# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# fixed = 1
# test_weights, test_biases = force_ones_fixed(lrr_models, fixed=fixed)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_fixed_({fixed}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)

# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models_04.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# fixed = 1
# test_weights, test_biases = force_ones_fixed(lrr_models, fixed=fixed)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_04_fixed_({fixed}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)


# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models_07.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# fixed = 1
# test_weights, test_biases = force_ones_fixed(lrr_models, fixed=fixed)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_07_fixed_({fixed}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)

# ------------------------------------------------------------------------
#  python runner2.py > RR_logs/log_force4.txt 2>&1
with open('RR_ckpt/LRR/llama8b/lrr_models.pkl', 'rb') as file:
    lrr_models = pickle.load(file)

fixed = 3
test_weights, test_biases = force_ones_fixed(lrr_models, fixed=fixed)

with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_fixed_({fixed}).pkl', 'wb') as file:
    pickle.dump((test_weights, test_biases), file)

# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models_04.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# fixed = 3
# test_weights, test_biases = force_ones_fixed(lrr_models, fixed=fixed)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_04_fixed_({fixed}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)

# # ------------------------------------------------------------------------
# with open('RR_ckpt/LRR/llama8b/lrr_models_07.pkl', 'rb') as file:
#     lrr_models = pickle.load(file)

# fixed = 3
# test_weights, test_biases = force_ones_fixed(lrr_models, fixed=fixed)

# with open(f'/scratch/bbjr/skarmakar/neuinv/min_rank/llama8b/W_b_07_fixed_({fixed}).pkl', 'wb') as file:
#     pickle.dump((test_weights, test_biases), file)
