import sys

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

# model_type = 'llama'
# MODEL_VERSION='3.1'
# MODEL_SIZE='8B'

# model_type = 'gemma'
# # MODEL_VERSION='2'
# MODEL_VERSION='3'
# # MODEL_SIZE='1B'
# # MODEL_SIZE='9B'
# MODEL_SIZE='12B'

model_type = 'qwen'
MODEL_VERSION='3'
MODEL_SIZE='4B'
# MODEL_SIZE='8B'



llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

hidden_layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))
print(hidden_layers)










# all_langs = ['english', 'german', 'hindi', 'spanish', 'french', 'italian', 'portuguese', 'thai', 'chinese_simplified', 'chinese_traditional', 'japanese', 'korean', 'russian', 'arabic', 'vietnamese', 'indonesian', 'turkish', 'polish', 'dutch', 'ukrainian', 'czech', 'romanian', 'greek', 'hungarian', 'swedish', 'danish', 'finnish', 'norwegian', 'bulgarian', 'serbian', 'croatian', 'slovak', 'lithuanian', 'slovenian', 'latvian', 'estonian', 'catalan', 'hebrew', 'persian', 'tagalog', 'bengali', 'urdu', 'tamil', 'telugu', 'malayalam', 'kannada', 'marathi', 'gujarati', 'punjabi', 'malay', 'swahili', ]

all_langs = ['english', 'german', 'hindi', 'spanish', 'french', 'italian', 'portuguese', 'thai', 'chinese_simplified', 'chinese_traditional', 'japanese', 'korean', 'russian', 'arabic', 'vietnamese', 'indonesian', 'turkish', 'polish', 'dutch', 'ukrainian', 'czech',]

# all_langs.reverse()

source_langs = ['english',]
target_langs = ['italian', ]

# test_set = ['portuguese', 'japanese',]

# other_langs = [element for element in all_langs if element not in source_langs + target_langs + test_set]
other_langs = [element for element in all_langs if element not in source_langs + target_langs]

# original_langs = ['english', 'french', 'german', 'hindi', 'italian', 'portuguese', 'spanish', 'thai']
# other_langs = ['english', 'french', 'german', 'hindi', 'italian', 'portuguese', 'spanish', 'thai']
# # other_langs = ['english', 'french', 'hindi']



# # dataset_label = 'language'
# dataset_label = 'language_multi'

# save_path = 'all_gitignore/language/'
# save_path = 'all_gitignore/language_multi/'


# save_path = 'all_gitignore/xRFM/language_multiex_llama/'
# save_path = 'all_gitignore/xRFM/language_multi_gemma/'
save_path = 'all_gitignore/xRFM/language_multiex_qwen/'

# save_path = 'all_gitignore/xRFM/test/'
# save_path = 'all_gitignore/xRFM/test_eigenpro/'

# save_path = 'all_gitignore/xRFM/language_multi/'


# METHOD = 'mean_difference'
METHOD = 'rfm'

solver = 'lstsq'
# solver = 'eigenpro'

for source_lang in source_langs:
    for other_lang in other_langs:
        print(f"============================================={source_lang} to {other_lang}=============================================")
        
        # if dataset_label == 'language':
        #     dataset = utils.language_dataset(llm, original_lang, other_lang)
        # elif dataset_label == 'language_multi':
        #     dataset = utils.multilingual_dataset(llm, original_lang, other_lang)

        dataset = utils.multilingual_dataset(llm, source_lang, other_lang)
        
        compute_save_directions_translate(llm, dataset, other_lang, control_method=METHOD, orig_lang=source_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

for target_lang in target_langs:
    for other_lang in other_langs:
        print(f"============================================={target_lang} to {other_lang}=============================================")

        dataset = utils.multilingual_dataset(llm, target_lang, other_lang)
        
        compute_save_directions_translate(llm, dataset, other_lang, control_method=METHOD, orig_lang=target_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break


# dest

for source_lang in other_langs:
    for other_lang in source_langs:
        print(f"============================================={source_lang} to {other_lang}=============================================")
        
        # if dataset_label == 'language':
        #     dataset = utils.language_dataset(llm, original_lang, other_lang)
        # elif dataset_label == 'language_multi':
        #     dataset = utils.multilingual_dataset(llm, original_lang, other_lang)

        dataset = utils.multilingual_dataset(llm, source_lang, other_lang)
        
        compute_save_directions_translate(llm, dataset, other_lang, control_method=METHOD, orig_lang=source_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

for target_lang in other_langs:
    for other_lang in target_langs:
        print(f"============================================={target_lang} to {other_lang}=============================================")

        dataset = utils.multilingual_dataset(llm, target_lang, other_lang)
        
        compute_save_directions_translate(llm, dataset, other_lang, control_method=METHOD, orig_lang=target_lang, solver=solver, path=save_path)
        
        del dataset
        torch.cuda.empty_cache()

        gc.collect()

    #     break
    # break

