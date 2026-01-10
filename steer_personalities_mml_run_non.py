import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor
# from transformers import Gemma3ForCausalLM, BitsAndBytesConfig
from neural_controllers import NeuralController
from utils import LLMType
from collections import namedtuple 
from tqdm import tqdm 
import gc
import os

# from janus.utils.io import load_pil_images
# from generation_utils import extract_image
import utils
import utils_concepts

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])


def select_llm(model_type, MODEL_VERSION='3.1', MODEL_SIZE='8B'):

    custom_cache_dir = "/scratch/bbjr/skarmakar/huggingface"

    if model_type=='llama':

        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_id = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda", cache_dir=custom_cache_dir,
        )

        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0 
        # model_name='llama_3_8b_it'
        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_name='llama_3_8b_it_eng_only'
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_name = "llama_3.1_70b_it_eng_only"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_name = "llama_3.3_70b_it_eng_only"

        processor = None
        llm_type = LLMType.TEXT

        language_model.generation_config.pad_token_id = tokenizer.pad_token_id # to disable the warning

        language_model.generation_config.temperature=None # to disable the stupid warnings
        language_model.generation_config.top_p=None # to disable the stupid warnings
        # language_model.generation_config.top_k=None # to disable the stupid warnings

    elif model_type=='gemma':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        if MODEL_VERSION == '3.1' and MODEL_SIZE == '1B':
            model_id = "google/gemma-3-1b-it"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        language_model = Gemma3ForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config
        ).eval()

        if MODEL_VERSION == '3.1' and MODEL_SIZE == '1B':
            model_name='gemma_3_1b_it_eng_only'

        processor = None
        llm_type = LLMType.GEMMA_TEXT

        # print(tokenizer.chat_template)

    llm = LLM(language_model, tokenizer, processor, model_name, llm_type)
    # print(llm.language_model)
    return llm


def compute_save_directions(llm, dataset, concept, control_method='rfm', path='directions/'):

    if os.path.exists(os.path.join(path, f'{control_method}_{concept}_{llm.name}.pkl')):
        print(f"'{os.path.join(path, f'{control_method}_{concept}_{llm.name}.pkl')}' exists, skipping it.")
        return

    concept_types = [concept]
    for concept_type in concept_types:
        controller = NeuralController(
            llm,
            llm.tokenizer,
            rfm_iters=8,
            control_method=control_method,
            n_components=1,
        )
        controller.compute_directions(dataset[concept_type]['train']['inputs'], dataset[concept_type]['train']['labels'])

        controller.save(concept=f'{concept_type}', model_name=llm.name, path=path)        



def read_file(fname, lower=True):

    concepts = []
    with open(fname, encoding="utf-8") as f: 
        for line in f:
            if lower:
                concepts.append(line.strip().lower())
            else:
                concepts.append(line.strip())
    concepts = sorted(list(set(concepts)))
    return concepts

def main():
    # /scratch/bbjr/skarmakar/neuinv/directions_dump
    
    torch.backends.cudnn.benchmark = True        
    torch.backends.cuda.matmul.allow_tf32 = True        

    # fnames = ['data/adjectives/social.txt',]
    # lowers = [True,]
    # dataset_labels = ['social',]
    # # save_path = 'all_gitignore/directions_single_new_statement_new_prompt/{}/'
    # # save_path = 'all_gitignore/directions_single_new_statement_old_prompt/{}/'
    # # save_path = 'all_gitignore/directions_single_old_statement_new_prompt/{}/'
    # save_path = 'all_gitignore/directions_single_old_statement_old_prompt/{}/'

    # -------------------------------------------------------------------------------
    fnames = ['data/adjectives/complexity.txt',
            'data/adjectives/logic.txt',
            'data/adjectives/physical.txt',
            'data/adjectives/social.txt',
            'data/adjectives/state.txt',
            'data/adjectives/texture.txt',
            'data/adjectives/time.txt',]
    lowers = [True, True, True, True, True, True, True, ]
    dataset_labels = ['complexity', 'logic', 'physical', 'social', 'state', 'texture', 'time']
    # save_path = 'all_gitignore/directions_adjectives_llama/{}/'
    save_path = 'all_gitignore/directions_adjectives_llama/{}/'


    model_type = 'llama'
    MODEL_VERSION='3.1'
    MODEL_SIZE='8B'
    llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

    # METHOD = 'mean_difference'
    METHOD = 'rfm'

    for f_idx, fname in enumerate(fnames):
        concepts = read_file(fname, lower=lowers[f_idx])
        # --------------------------
        # concepts = ["cheerful", "gloomy",]
        # --------------------------
        dataset_label = dataset_labels[f_idx]

        folder_path = save_path.format(dataset_label)
        os.makedirs(folder_path, exist_ok=True)

        for concept_idx, concept in enumerate(tqdm(concepts)):
            print(f"=============================================CONCEPT={concept}=============================================")

            dataset = utils_concepts.pca_adjective_dataset(llm, concept, dataset_label)
            
            # if dataset_label == 'complexity':
            #     dataset = utils_concepts.pca_fears_dataset(llm, concept)
            # elif dataset_label == 'logic':
            #     dataset = utils_concepts.pca_personalities_dataset(llm, concept)
            # elif dataset_label == 'physical':
            #     dataset = utils_concepts.pca_persona_dataset(llm, concept)
            # elif dataset_label == 'social':
            #     dataset = utils_concepts.pca_mood_dataset(llm, concept)
            # elif dataset_label == 'state':
            #     dataset = utils_concepts.pca_concept_dataset(llm, concept)
            # elif dataset_label == 'texture':
            #     dataset = utils_concepts.pca_places_dataset(llm, concept)
            # elif dataset_label == 'time':
            #     dataset = utils_concepts.pca_places_dataset(llm, concept)


            compute_save_directions(llm, dataset, concept, control_method=METHOD, path=folder_path)
            # del llm
            del dataset
            torch.cuda.empty_cache()

        gc.collect()



if __name__ == "__main__":
    main()