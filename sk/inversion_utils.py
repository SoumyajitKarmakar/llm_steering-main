import sys
import os
from pathlib import Path

notebook_path = Path().absolute()
sys.path.append(str(notebook_path.parent))



import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import LLMType

import matplotlib.pyplot as plt
import seaborn as sns

from neural_controllers import NeuralController
from collections import namedtuple

# from scipy.linalg import eigh
from scipy.linalg import eig
from scipy.stats import entropy
import math

from scipy.linalg import inv

from tqdm import tqdm


from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

import re
import logging


LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])

def select_llm(model_type, MODEL_VERSION='3.1', MODEL_SIZE='8B', base=False):

    custom_cache_dir = "/scratch/bbjr/skarmakar/huggingface"

    if model_type=='llama':

        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            if base:
                model_id = "meta-llama/Llama-3.1-8B"
                print(f"Loading {model_id}")
            else:
                model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                print(f"Loading {model_id}")
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
            if base:
                model_name='llama_3_8b_eng_only'
            else:
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
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        if MODEL_VERSION == '3' and MODEL_SIZE == '1B':
            model_id = "google/gemma-3-1b-it"
        elif MODEL_VERSION == '3' and MODEL_SIZE == '12B':
            model_id = "google/gemma-3-12b-it"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # language_model = Gemma3ForCausalLM.from_pretrained(
        #     model_id, quantization_config=quantization_config
        # ).eval()

        language_model = Gemma3ForCausalLM.from_pretrained(
            model_id, device_map="cuda", torch_dtype="auto", cache_dir=custom_cache_dir,
        ).eval()

        if MODEL_VERSION == '3' and MODEL_SIZE == '1B':
            model_name='gemma_3_1b_it_eng_only'
        elif MODEL_VERSION == '3' and MODEL_SIZE == '12B':
            model_name='gemma_3_12b_it_eng_only'

        processor = None
        llm_type = LLMType.GEMMA_TEXT

        # print(tokenizer.chat_template)

    elif model_type=='qwen':

        if MODEL_VERSION == '3' and MODEL_SIZE == '8B':
            model_id = "Qwen/Qwen3-8B"

        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda", torch_dtype="auto", cache_dir=custom_cache_dir,
        )


        # use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        # tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = 0

        if MODEL_VERSION == '3' and MODEL_SIZE == '8B':
            model_name='qwen_3_8b_eng_only'

        processor = None
        llm_type = LLMType.TEXT

        language_model.generation_config.pad_token_id = tokenizer.pad_token_id # to disable the warning

        language_model.generation_config.temperature=None # to disable the stupid warnings
        language_model.generation_config.top_p=None # to disable the stupid warnings
        language_model.generation_config.top_k=None # to disable the stupid warnings

    llm = LLM(language_model, tokenizer, processor, model_name, llm_type)
    # print(llm.language_model)
    return llm


def compare_pearson(concept1, concept2):
    total_v = 0
    for l in concept1:
        mat = torch.stack((concept1[l][0], concept2[l][0]))
        
        pearson = torch.corrcoef(mat)
        v = pearson[0, 1].item()
        print(f'layer: {l}, PCC: {v:.2f}')
        total_v += v
    
    avg = total_v / len(concept1)
    print(f'Average: {avg:.2f}')
    
    return avg


def compare_cosine(concept1, concept2):
    total_c = 0
    for l in concept1:
        cosine = torch.nn.functional.cosine_similarity(concept1[l][0], concept2[l][0], dim=0)
        c = cosine.item()

        print(f'layer: {l}, cosine: {c}')
        total_c += c
        
    print(f'total: {total_c}')
    return total_c


def just_dirs(llm, concept, path='../all_gitignore/directions_moods/'):
    tcontroller = NeuralController(
        llm,
        llm.tokenizer,
        rfm_iters=8,
        control_method="rfm",
        n_components=1
    )

    tcontroller.load(concept=concept, model_name=llm.name, path=path)

    return tcontroller.directions


def load_controller(llm, concept, path='../all_gitignore/directions_moods/'):
    tcontroller = NeuralController(
        llm,
        llm.tokenizer,
        rfm_iters=8,
        control_method="rfm",
        n_components=1,
    )

    tcontroller.load(concept=concept, model_name=llm.name, path=path)

    return tcontroller

def clean_llama(s):
    if isinstance(s, str):
        s1 = s.replace("<|start_header_id|>assistant<|end_header_id|>\n", "\n-----------------------------------------------------")
        s2 = re.sub(r'<[^>]*>', '', s1)

        u = s2.find("user")

        s3 = s2[u+6:]

        return s3
    elif isinstance(s, list):
        c_s = []
        
        for i in s:
            s1 = i.replace("<|start_header_id|>assistant<|end_header_id|>\n", "\n-----------------------------------------------------")
            s2 = re.sub(r'<[^>]*>', '', s1)

            u = s2.find("user")

            s3 = s2[u+6:]

            c_s.append(s3)
        
        return c_s

def test_concept_vector(controller, concept="___", prompts=[], coef=0.75, max_tokens=100, orig=True, image=None, qwen=False):

    outputs = []
    for prompt in prompts:
        if orig:
            print("\n========================== No Control ==========================")
            if qwen:
                original_output = controller.generate_qwen(prompt, image=image, max_new_tokens=max_tokens, do_sample=False)
            else:
                original_output = controller.generate(prompt, image=image, max_new_tokens=max_tokens, do_sample=False)#, temperature=0)
            
            print(clean_llama(original_output))

        print(f"\n========================== + {concept} Control (normal) ==========================")
        if qwen:
            steered_output = controller.generate_qwen(prompt,
                                                image=image, 
                                                layers_to_control=controller.hidden_layers,
                                                control_coef=coef,
                                                max_new_tokens=max_tokens,
                                                do_sample=False)
        else:
            steered_output = controller.generate(prompt,
                                                image=image, 
                                                layers_to_control=controller.hidden_layers,
                                                control_coef=coef,
                                                max_new_tokens=max_tokens,
                                                do_sample=False)
        
        print(clean_llama(steered_output))

        outputs.append(steered_output)

        torch.cuda.empty_cache()

    return outputs

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



def find_lingering_forward_hooks(model):
    lingering_hooks = []
    for name, module in model.named_modules():
        if module._forward_hooks:
            hook_info = (name, 'forward')
            lingering_hooks.append(hook_info)
            print(f"Warning: Found {len(module._forward_hooks)} lingering 'forward' hook(s) on module: {name}")

        if module._forward_pre_hooks:
            hook_info = (name, 'forward_pre')
            lingering_hooks.append(hook_info)
            print(f"Warning: Found {len(module._forward_pre_hooks)} lingering 'forward_pre' hook(s) on module: {name}")
            
    if not lingering_hooks:
        print("Success: No lingering forward hooks found in the model.")
        
    return lingering_hooks



def read_tuples_as_list(llm, antonyms, path='../directions_moods/'):
    hidden_layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))

    Xt = []
    Yt = []

    for t in antonyms:
        dir1 = just_dirs(llm, t[0], path=path)
        dir2 = just_dirs(llm, t[1], path=path)

        Xt.append(dir1)
        Yt.append(dir2)

    return Xt, Yt


def read_tuples(llm, antonyms, path='../directions_moods/'):
    hidden_layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))

    Xt = {i: [] for i in hidden_layers}
    Yt = {i: [] for i in hidden_layers}

    for t in antonyms:
        dir1 = just_dirs(llm, t[0], path=path)
        dir2 = just_dirs(llm, t[1], path=path)

        for k in Xt:
            Xt[k].append(dir1[k])
            Yt[k].append(dir2[k])

    X = {i: torch.cat(Xt[i]).to("cuda") for i in hidden_layers}
    Y = {i: torch.cat(Yt[i]).to("cuda") for i in hidden_layers}

    return X, Y

def LRR_auto(X, Y):
    # add args for cv, and alpha
    lrr_models = {}
    
    d = X[-1].shape[1]

    for i in X:
        x = X[i].cpu().numpy()
        y = Y[i].cpu().numpy()

        # alphas = 10.0 ** np.arange(-1, 6)  # log grid
        # reg_lrr = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=10))

        reg_lrr = make_pipeline(StandardScaler(), Ridge(alpha=10000.0, solver='cholesky'))
        
        model_lrr = TransformedTargetRegressor(regressor=reg_lrr, transformer=StandardScaler())

        model_lrr.fit(x, y)

        # best_alpha_lrr = model_lrr.regressor_.named_steps["ridgecv"].alpha_
        # print(f"Layer: {i}, best lambda: {best_alpha_lrr}")

        print(f"Layer {i} done.")

        # xtx = x.T @ x
        # A = torch.linalg.solve(xtx + lambda_reg * torch.eye(d).to("cuda"), x.T @ y) # switch to more robust

        lrr_models[i] = model_lrr
    
    return lrr_models

def KRR_auto(X, Y):
    # add args for cv, gamma, and alpha
    krr_models = {}
    
    d = X[-1].shape[1]

    for i in X:
        x = X[i].cpu().numpy()
        y = Y[i].cpu().numpy()

        reg_krr = make_pipeline(StandardScaler(), KernelRidge(kernel="laplacian"))
        param_grid = {
            "kernelridge__alpha": 10.0 ** np.arange(-5, 1),
            # optionally tune gamma; default is 1/n_features (~2.44e-4 for 4096)
            "kernelridge__gamma": 10.0 ** np.arange(-6, 0)
        }
        search_krr = GridSearchCV(reg_krr, param_grid=param_grid, cv=10, scoring="neg_mean_squared_error")
        model_krr = TransformedTargetRegressor(regressor=search_krr, transformer=StandardScaler())
        model_krr.fit(x, y)

        best_alpha_krr = model_krr.regressor_.best_params_["kernelridge__alpha"]
        best_gamma_krr = model_krr.regressor_.best_params_.get("kernelridge__gamma")

        print(f"Layer: {i}, best lambda: {best_alpha_krr}, best gamma: {best_gamma_krr}")

        krr_models[i] = model_krr
    
    return krr_models

def apply_auto(direction, models, layers_control=None):
    new_predicted = {}

    if layers_control is None: 
        for i in direction:
            # new_predicted[i] = direction[i] @ lrr_models[i]
            new_predicted[i] = torch.tensor(models[i].predict(direction[i].cpu().numpy())).to("cuda")
    
    else:
        for i in direction:
            if i in layers_control:
                # print(f"control {i}")
                # new_predicted[i] = direction[i] @ lrr_models[i]
                new_predicted[i] = torch.tensor(models[i].predict(direction[i].cpu().numpy())).to("cuda")
            else:
                # new_predicted[i] = direction[i]
                new_predicted[i] = torch.zeros_like(direction[i])
    
    return new_predicted


# def LRR(llm, antonyms, lambda_reg=0.1, d = 4096, path='../directions_moods/'):
#     # Linear Ridge Regression
    
#     hidden_layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))

#     Xt = {i: [] for i in hidden_layers}
#     Yt = {i: [] for i in hidden_layers}

#     for t in antonyms:
#         dir1 = just_dirs(llm, t[0], path=path)
#         dir2 = just_dirs(llm, t[1], path=path)

#         for k in Xt:
#             Xt[k].append(dir1[k])
#             Yt[k].append(dir2[k])

#     X = {i: torch.cat(Xt[i]) for i in hidden_layers}
#     Y = {i: torch.cat(Yt[i]) for i in hidden_layers}

def LRR(X, Y, lambda_reg=0.1):

    trans_mat = {i: [] for i in X}
    
    d = X[-1].shape[1]

    for i in trans_mat:
        x = X[i].to("cuda")
        y = Y[i].to("cuda")

        xtx = x.T @ x
        A = torch.linalg.solve(xtx + lambda_reg * torch.eye(d).to("cuda"), x.T @ y) # switch to more robust

        trans_mat[i] = A
    
    return trans_mat

# def apply_lrr(direction, trans_mat):
#     new_predicted = {i: [] for i in direction}

#     for i in new_predicted:
#         new_predicted[i] = direction[i] @ trans_mat[i]
    
#     return new_predicted

def apply_lrr(direction, trans_mat, biases=None):
    new_predicted = {i: [] for i in direction}

    for i in new_predicted:
        if biases is None:
            new_predicted[i] = direction[i] @ trans_mat[i]
        else:
            new_predicted[i] = direction[i] @ trans_mat[i] + biases[i]
    
    return new_predicted

def apply_lrr_norm(direction, trans_mat, X_mean, X_std, Y_mean, Y_std):
    new_predicted = {i: [] for i in direction}

    for i in new_predicted:

        norm_dir = ((direction[i] - X_mean[i]) / X_std[i]).cpu().numpy()

        norm_pred = (norm_dir @ trans_mat[i]).to("cuda")

        new_predicted[i] = norm_pred * Y_std[i] + Y_mean[i]
    
    return new_predicted


def apply_krr(direction, trans_krr):
    new_predicted = {i: [] for i in direction}

    for i in new_predicted:
        # new_predicted[i] = direction[i] @ trans_mat[i]

        new_predicted[i] = torch.tensor(trans_krr[i].predict(direction[i].cpu().numpy().reshape(1, -1))).to("cuda")
    
    return new_predicted


def apply_krr_norm(direction, trans_krr, X_mean, X_std, Y_mean, Y_std):
    # norm_direction = {i: (direction[i] - X_mean[i]) / X_std[i] for i in direction}

    new_predicted = {}

    for i in direction:
        # new_predicted[i] = direction[i] @ trans_mat[i]

        # norm_dir = ((direction[i] - X_mean[i]) / X_std[i]).cpu().numpy().reshape(1, -1)
        norm_dir = ((direction[i] - X_mean[i]) / X_std[i]).cpu().numpy()

        norm_pred = torch.tensor(trans_krr[i].predict(norm_dir)).to("cuda")

        # new_predicted[i] = {i: lrr_norm_predicted5[i] * X_std[i] + X_mean[i] for i in temp}
        new_predicted[i] = norm_pred * Y_std[i] + Y_mean[i]
    
    return new_predicted

# def KRR(llm, antonyms, kernel="laplacian", lambda_reg=0.1, gamma=0.001, path='../directions_moods/'): # todo update it
#     # Kernel Ridge Regression
#     hidden_layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))

#     Xt = {i: [] for i in hidden_layers}
#     Yt = {i: [] for i in hidden_layers}

#     for t in antonyms:
#         dir1 = just_dirs(llm, t[0], path=path)
#         dir2 = just_dirs(llm, t[1], path=path)

#         for k in Xt:
#             Xt[k].append(dir1[k])
#             Yt[k].append(dir2[k])

#     X = {i: torch.cat(Xt[i]) for i in hidden_layers}
#     Y = {i: torch.cat(Yt[i]) for i in hidden_layers}


def KRR(X, Y, kernel="laplacian", lambda_reg=0.1, gamma=0.001):

    trans_krr = {i: [] for i in X}
    # d = 4096

    for i in trans_krr:
        x = X[i].cpu()
        y = Y[i].cpu()

        # Kernel Ridge Regression with Laplace kernel
        krr = KernelRidge(
            kernel=kernel,              # Laplace kernel
            alpha=lambda_reg,           # Regularization parameter
            gamma=gamma,                 # Kernel bandwidth (1/sigma)
        )

        # Fit the model - sklearn supports multi-output regression
        krr.fit(x.numpy(), y.numpy())

        trans_krr[i] = krr

    return trans_krr


def get_W_b(model_lrr):
    # Fitted objects
    pipe = model_lrr.regressor_                       # the fitted Pipeline
    try:
        ridge = pipe.named_steps["ridgecv"]               # the fitted RidgeCV
    except:
        ridge = pipe.named_steps["ridge"]
    scaler_x = pipe.named_steps["standardscaler"]     # X StandardScaler
    scaler_y = model_lrr.transformer_                 # y StandardScaler (from TTR)

    # Coefficients in standardized space
    # ridge.coef_.shape == (n_targets, n_features) == (4096, 4096)
    C = ridge.coef_           # in standardized X, standardized y space
    b = ridge.intercept_      # length 4096 (in standardized y space)

    # Build effective W and intercept in original units
    sx = scaler_x.scale_      # length 4096
    mx = scaler_x.mean_       # length 4096
    sy = scaler_y.scale_      # length 4096
    my = scaler_y.mean_       # length 4096

    # W_eff: 4096 x 4096
    W_eff = (C.T / sx[:, None]) * sy[None, :]

    # b_eff: length 4096
    b_eff = sy * b - (mx / sx) @ C.T * sy + my

    return W_eff, b_eff

def force_ones(models, thresh=0.1):
    new_models = {}
    biases = {}

    for layer in tqdm(models):
        weight, bias = get_W_b(models[layer])

        # eigen_w, eigen_v = eigh(weight, check_finite=True) # wrong: not symmetric
        eigen_w, eigen_v = eig(weight, check_finite=True)

        for i in eigen_w:
            if np.abs(i.imag) > 0.01:
                print(f"First Imaginary problem: {i}")

        eigen_w_new = [1.0 if i.real > thresh else -1.0 if i.real <-1*thresh else 0.0 for i in eigen_w]

        new_mo = eigen_v @ np.diag(eigen_w_new) @ inv(eigen_v)
        for i in new_mo:
            if np.max(np.abs(i.imag)) > 0.01:
                print(f"Second Imaginary problem: {i}")

        new_models[layer] = torch.tensor(new_mo.real).to(device='cuda', dtype=torch.float32)
        biases[layer] = torch.tensor(bias).to(device='cuda', dtype=torch.float32)

    return new_models, biases

def force_ones_fixed(models, fixed=5):
    print(f"Runing with fixed={fixed}")
    new_models = {}
    biases = {}

    for layer in tqdm(models):
        weight, bias = get_W_b(models[layer])

        # eigen_w, eigen_v = eigh(weight, check_finite=True) # wrong: not symmetric
        eigen_w, eigen_v = eig(weight, check_finite=True)

        for i in eigen_w:
            if np.abs(i.imag) > 0.01:
                print(f"First Imaginary problem: {i}")

        eigen_w_new = np.zeros_like(eigen_w, dtype=float)

        sort_idxs = np.argsort(eigen_w.real)

        eigen_w_new[sort_idxs[-fixed:]] = 1.0
        eigen_w_new[sort_idxs[:fixed]] = -1.0

        new_mo = eigen_v @ np.diag(eigen_w_new) @ inv(eigen_v)

        for i in new_mo:
            if np.max(np.abs(i.imag)) > 0.01:
                print(f"Second Imaginary problem: {i}")

        new_models[layer] = torch.tensor(new_mo.real).to(device='cuda', dtype=torch.float32)
        biases[layer] = torch.tensor(bias).to(device='cuda', dtype=torch.float32)

    return new_models, biases





def get_heatmap(trans_mat, annot=False, save_path=None):

    t_data = []
    layers = []

    for i in trans_mat:
        t_data.append(torch.flatten(trans_mat[i]))
        layers.append(i)

    heat_data = torch.stack(t_data)
    # Compute correlation matrix using PyTorch
    # Shape will be (30, 30) - correlation between each pair of vectors
    corr_matrix = torch.corrcoef(heat_data)

    # Convert to numpy for plotting
    corr_matrix_np = corr_matrix.cpu().numpy()

    # # Create the heatmap
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix_np, 
    #             cmap='coolwarm',  # Blue-white-red colormap
    #             center=0,          # Center colormap at 0
    #             vmin=-1, vmax=1,   # Correlation range
    #             square=True,       # Square cells
    #             linewidths=0.5)    # Grid lines

    # plt.title('Correlation Heatmap of 30 Vectors')
    # plt.tight_layout()
    # plt.show()

    # -------------------------------
    # Generate mask for upper triangle
    # mask = np.triu(np.ones_like(corr_matrix_np, dtype=bool), k=1)
    mask = np.triu(np.ones_like(corr_matrix_np, dtype=bool))

    # Create triangular heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix_np,
                xticklabels=layers,
                yticklabels=layers,
                annot=annot,              # Show correlation values
                fmt='.2f',
                mask=mask,               # Hide upper triangle
                cmap=sns.diverging_palette(230, 20, as_cmap=True),
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8})

    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def spectral_indicators(M, tau=0.9, eps=1e-12, clip_neg=False):
    # w, V = eigh(M, lower=True, check_finite=False) # care, eigh is only for symmetric matrix
    w, V = eig(M, lower=True, check_finite=False) # care, eigh is only for symmetric matrix
    
    if clip_neg:
        w = np.clip(w, 0.0, None)
    
    lam = np.flip(w)
    
    tr = lam.sum()
    lam2_sum = np.dot(lam, lam)
    lam_max = lam[0] if lam.size > 0 else 0.0
    lam_min = lam[-1] if lam.size > 0 else 0.0

    # Condition number (2-norm)
    cond2 = (lam_max / max(lam_min, eps)) if lam.size > 0 else np.nan

    # Stable rank: ||M||_F^2 / ||M||_2^2 = (sum lam_i^2) / lam_max^2
    stable_rank = (lam2_sum / max(lam_max**2, eps)) if lam_max > 0 else 0.0

    # Participation ratio: (sum lam)^2 / sum lam^2
    pr = (tr**2 / max(lam2_sum, eps)) if lam2_sum > 0 else 0.0

    # # Entropic effective rank (eRank)
    # if tr > 0:
    #     p = lam / tr
    #     p_safe = np.where(p > 0, p, 1.0)
    #     H = -np.sum(p * np.log(p_safe))
    #     erank = float(np.exp(H))
    # else:
    #     erank = 0.0

    # Entropic effective rank (eRank)
    if tr > 0:
        p = lam / tr
        H = entropy(p, base=np.e)
        erank = float(np.exp(H))
    else:
        erank = 0.0

    # Cumulative energy and tau-based k
    if tr > 0:
        cum = np.cumsum(lam) / tr
        k_tau = int(np.searchsorted(cum, tau) + 1)
        cumulative_energy = cum
    else:
        k_tau = 0
        cumulative_energy = np.zeros_like(lam)

    return {
        "eigenvalues_desc": lam,
        "condition_number_2": float(cond2),
        "stable_rank": float(stable_rank),
        "participation_ratio": float(pr),
        "effective_rank_entropy": float(erank),
        "cumulative_energy": cumulative_energy,
        "k_at_tau": k_tau,
        "eigenvectors": V[:, ::-1],  # columns match descending lam
    }

def find_min_ind(l, v):
    for i in range(len(l)):
        if l[i] < v:
            return i


# def setup_logger(log_file='training.log'):
#     """
#     Setup logger to write to file and console
#     """
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()  # Also print to console
#         ]
#     )
