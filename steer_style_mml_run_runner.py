import sys
# import os
# from pathlib import Path

notebook_path = "/u/skarmakar1/version_check/llm_steering-main/sk"
sys.path.append(notebook_path)

from inversion_utils import *

import torch
import numpy as np

import utils

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



llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)



hidden_layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))
print(hidden_layers)


all_authors = ["hemingway", "faulkner", "joyce", "austen", "woolf", "kafka", "stein", "cummings", "nabokov", "wilde", "twain", "wodehouse", "mccarthy", "carver", "bukowski", "márquez", "borges", "thompson", "lovecraft", "pratchett",]


source_authors = ['kafka',]
target_authors = ['hemingway',]


other_authors = [element for element in all_authors if element not in source_authors + target_authors]


save_path = 'all_gitignore/xRFM/style_llama/'

METHOD = 'rfm'

solver = 'lstsq'



for source_author in source_authors:
    for other_author in other_authors:
        print(f"============================================={source_author} to {other_author}=============================================")

        dataset = utils.literary_style_dataset(llm, concept_type=other_author, concept_source=source_author)
        
        compute_save_directions_translate(llm, dataset, other_author, control_method=METHOD, orig_lang=source_author, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

for target_author in target_authors:
    for other_author in other_authors:
        print(f"============================================={target_author} to {other_author}=============================================")

        dataset = utils.literary_style_dataset(llm, concept_type=other_author, concept_source=target_author)
        
        compute_save_directions_translate(llm, dataset, other_author, control_method=METHOD, orig_lang=target_author, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break


for other_author in other_authors:
    for source_author in source_authors:
        print(f"============================================={other_author} to {source_author}=============================================")
        
        # dataset = utils.multilingual_dataset(llm, other_author, source_author)
        dataset = utils.literary_style_dataset(llm, concept_type=source_author, concept_source=other_author)
        
        compute_save_directions_translate(llm, dataset, source_author, control_method=METHOD, orig_lang=other_author, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

for other_author in other_authors:
    for target_author in target_authors:
        print(f"============================================={other_author} to {target_author}=============================================")

        # dataset = utils.multilingual_dataset(llm, other_author, target_author)
        dataset = utils.literary_style_dataset(llm, concept_type=target_author, concept_source=other_author)
        
        compute_save_directions_translate(llm, dataset, target_author, control_method=METHOD, orig_lang=other_author, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

    