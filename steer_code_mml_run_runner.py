import sys

notebook_path = "/u/skarmakar1/version_check/llm_steering-main/sk"
sys.path.append(notebook_path)

from inversion_utils import *
import utils

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



llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

hidden_layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))
print(hidden_layers)




all_prog_langs = ['python', 'javascript', 'java', 'c++', 'c#', 'c', 'typescript', 'php', 'golang', 'rust', 'swift', 'kotlin', 'r', 'ruby', 'dart', 'matlab', 'assembly', 'vba', 'shell', 'delphi']


source_prog_langs = ['python',]
target_prog_langs = ['c',]


other_prog_langs = [element for element in all_prog_langs if element not in source_prog_langs + target_prog_langs]


save_path = 'all_gitignore/xRFM/code_llama/'

METHOD = 'rfm'

solver = 'lstsq'

for source_prog_lang in source_prog_langs:
    for other_prog_lang in other_prog_langs:
        print(f"============================================={source_prog_lang} to {other_prog_lang}=============================================")

        dataset = utils.programming_style_dataset(llm, other_prog_lang, concept_source=source_prog_lang)
        
        compute_save_directions_translate(llm, dataset, other_prog_lang, control_method=METHOD, orig_lang=source_prog_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

for target_prog_lang in target_prog_langs:
    for other_prog_lang in other_prog_langs:
        print(f"============================================={target_prog_lang} to {other_prog_lang}=============================================")

        dataset = utils.programming_style_dataset(llm, other_prog_lang, concept_source=target_prog_lang)
        
        compute_save_directions_translate(llm, dataset, other_prog_lang, control_method=METHOD, orig_lang=target_prog_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break


for other_prog_lang in other_prog_langs:
    for source_prog_lang in source_prog_langs:
        print(f"============================================={other_prog_lang} to {source_prog_lang}=============================================")
        
        # dataset = utils.multilingual_dataset(llm, other_prog_lang, source_prog_lang)
        dataset = utils.programming_style_dataset(llm, source_prog_lang, concept_source=other_prog_lang)
        
        compute_save_directions_translate(llm, dataset, source_prog_lang, control_method=METHOD, orig_lang=other_prog_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

for other_prog_lang in other_prog_langs:
    for target_prog_lang in target_prog_langs:
        print(f"============================================={other_prog_lang} to {target_prog_lang}=============================================")

        # dataset = utils.multilingual_dataset(llm, other_prog_lang, target_prog_lang)
        dataset = utils.programming_style_dataset(llm, target_prog_lang, concept_source=other_prog_lang)
        
        compute_save_directions_translate(llm, dataset, target_prog_lang, control_method=METHOD, orig_lang=other_prog_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

