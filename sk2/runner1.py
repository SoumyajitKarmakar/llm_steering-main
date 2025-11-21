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


#  python runner1.py >> ../all_gitignore/sk2_items/RR_logs/log_seed1.txt 2>&1

# SEED = 0
SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True

LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])


model_type = 'llama'
# model_type = 'qwen'

# MODEL_VERSION = '3'
MODEL_VERSION = '3.1'
# MODEL_VERSION = '3.3'

MODEL_SIZE = '8B'
# MODEL_SIZE = '70B'

llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

with open("../data/moods/all_antonym_pairs.pkl", 'rb') as file:
    all_e = pickle.load(file)



test_size = 0.1

print("Total data:", len(all_e))
print(all_e[:5])

train_data_t, test_data = train_test_split(all_e, test_size=test_size, random_state=SEED)

print("Training data normal:", len(train_data_t))
print(train_data_t[:5])

swap_train_data = [(b, a) for a, b in train_data_t]
print("Training data swapped:", len(swap_train_data))
print(swap_train_data[:5])

train_data = train_data_t + swap_train_data
print("Training data:", len(train_data))
print(train_data[:5])

print("Testing data:", len(test_data))
print(test_data[:5])


# leak checking

list1 = []
for i in train_data:
    list1.append(i[0])
    list1.append(i[1])

list2 = []
for j in test_data:
    list2.append(j[0])
    list2.append(j[1])

set1 = set(list1)
set2 = set(list2)

print(len(set1))
print(set1)
print(len(set2))
print(set2)
print("*"*50)
print(set1.intersection(set2))
print(set2.intersection(set1))
print("aggressive" in set1)
print("aggressive" in set2)


X_train, Y_train = read_tuples(llm, train_data, path='../all_gitignore/directions_moods_plus_llama/')
X_test, Y_test = read_tuples(llm, test_data, path='../all_gitignore/directions_moods_plus_llama/')

print(test_data)


lrr_models = LRR_auto(X_train, Y_train)

with open(f'../all_gitignore/sk2_items/RR_ckpt/LRR/llama8b/lrr_models_S{SEED}.pkl', 'wb') as file:
    pickle.dump(lrr_models, file)

