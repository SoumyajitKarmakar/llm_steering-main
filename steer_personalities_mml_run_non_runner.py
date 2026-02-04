import sys

notebook_path = "/u/skarmakar1/version_check/llm_steering-main/sk"
sys.path.append(notebook_path)

from inversion_utils import *
import utils
import utils_concepts

import torch
import numpy as np

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True

model_type = 'llama'
MODEL_VERSION='3.1'
MODEL_SIZE='8B'

# model_type = 'gemma'
# # MODEL_VERSION='2'
# MODEL_VERSION='3'
# # MODEL_SIZE='1B'
# # MODEL_SIZE='9B'
# MODEL_SIZE='12B'

# model_type = 'qwen'
# MODEL_VERSION='3'
# MODEL_SIZE='4B'
# # MODEL_SIZE='8B'
# # MODEL_SIZE='30B'



llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

hidden_layers = list(range(-llm.language_model.config.num_hidden_layers+1, 0))
print(hidden_layers)

# Training

fnames = ['data/adjectives/social.txt',
        'data/adjectives/complexity.txt',
        'data/adjectives/logic.txt',
        'data/adjectives/physical.txt',
        'data/adjectives/state.txt',
        'data/adjectives/texture.txt',
        'data/adjectives/time.txt',]
lowers = [True, True, True, True, True, True, True, ]
dataset_labels = ['social', 'complexity', 'logic', 'physical', 'state', 'texture', 'time']

# save_path = 'all_gitignore/xRFM/directions_adjectives_qwen/{}/'
# save_path = 'all_gitignore/xRFM/directions_adjectives_gemma/{}/'
# save_path = 'all_gitignore/xRFM/directions_adjectives_gemma2/{}/'

# save_path = 'all_gitignore/xRFM/test/new_class0'
# save_path = 'all_gitignore/xRFM/test/new_class1'
# save_path = 'all_gitignore/xRFM/test/old_class0'
save_path = 'all_gitignore/xRFM/test/old_class1'

solver = 'lstsq'
# solver = 'eigenpro'

# data_file = "class0-1"
# data_file = "class0"
data_file = "class1"

# data_path = 'data/general_statements_adj/{}/'
data_path = 'data/general_statements/'


# METHOD = 'mean_difference'
METHOD = 'rfm'

for f_idx, fname in enumerate(fnames):
    # --------------------------
    if fname != 'data/adjectives/social.txt':
        continue
    # --------------------------

    concepts = read_file(fname, lower=lowers[f_idx])

    # --------------------------
    # concepts = ["arcane",]
    # concepts = ["brave",]
    # --------------------------
    
    dataset_label = dataset_labels[f_idx]

    folder_path = save_path.format(dataset_label)
    os.makedirs(folder_path, exist_ok=True)

    for concept_idx, concept in enumerate(tqdm(concepts)):
        print(f"=============================================CONCEPT={concept}=============================================")

        dataset = utils_concepts.pca_adjective_dataset(llm, concept, dataset_label, data_path=data_path, data_file=data_file)

        compute_save_directions(llm, dataset, concept, control_method=METHOD, solver=solver, path=folder_path)
        
        del dataset
        torch.cuda.empty_cache()

    #     break

    # break

    gc.collect()

