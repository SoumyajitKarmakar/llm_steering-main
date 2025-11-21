
# %%
import torch
import numpy as np

from inversion_utils import *
import pickle
from sklearn.model_selection import train_test_split

# %%
SEED = 0

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True

LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])

# %%
model_type = 'llama'
# model_type = 'qwen'

# MODEL_VERSION = '3'
MODEL_VERSION = '3.1'
# MODEL_VERSION = '3.3'

MODEL_SIZE = '8B'
# MODEL_SIZE = '70B'

llm = select_llm(model_type, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

# %%


# %% [markdown]
# ODE

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from copy import deepcopy
import logging

# %%
with open("../data/moods/all_antonym_pairs.pkl", 'rb') as file:
    all_e = pickle.load(file)

# %%
with open("inversion_matrices/llama8b/big_trans_mat.pkl", 'rb') as file:
    lrr = pickle.load(file)

# %%
str_X = [i[0] for i in all_e]
str_Y = [i[1] for i in all_e]

# %%
print(str_X)
print(len(str_X))
print(str_Y)

# %%
path = '../directions_moods_plus_llama/'

X = [just_dirs(llm, i, path=path) for i in str_X]
Y = [just_dirs(llm, i, path=path) for i in str_Y]

# %%
print(len(X))
print(X[10].keys())
print(X[10][-10].shape)

# %%
def setup_logger(log_file='training.log'):
    """
    Setup logger to write to file and console
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

# %%
class TransformDataset(Dataset):
    def __init__(self, inputs, outputs):
        """
        inputs: numpy array or tensor of shape (n_samples, 4096)
        outputs: numpy array or tensor of shape (n_samples, 4096)
        """
        self.inputs = inputs
        self.outputs = outputs
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# %%
# ODE Function
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=4096):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, t, y):
        return self.net(y)

# Neural ODE Model
class NeuralODEModel(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096):
        super(NeuralODEModel, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Encode input
        h = self.encoder(x)
        
        # Solve ODE from t=0 to t=1
        t = torch.tensor([0.0, 1.0]).to(x.device)
        h = odeint(self.ode_func, h, t, method='dopri5')[1]
        
        # Decode output
        output = self.decoder(h)
        return output

# %%
def cosine_loss(predictions, targets):
    return 1 - F.cosine_similarity(predictions, targets, dim=1).mean()

# %%
def train_model(model, train_loader, val_loader=None, num_epochs=5, lr=0.001, print_every=1, log_file='training.log'):
    
    setup_logger(log_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Track best validation loss and model state
    best_val_loss = float('inf')
    best_model_state = None

    logging.info(f'Starting training on device: {device}')
    logging.info(f'Number of epochs: {num_epochs}, Learning rate: {lr}')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            # loss = criterion(predictions, batch_outputs)
            loss = cosine_loss(predictions, batch_outputs)
            
            # print(f"train loss {loss}")
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase (if validation loader provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_inputs, batch_outputs in val_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_outputs = batch_outputs.to(device)
                    
                    predictions = model(batch_inputs)
                    # loss = criterion(predictions, batch_outputs)
                    loss = cosine_loss(predictions, batch_outputs)

                    # print(f"val loss {loss}")

                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Save best model state
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())
            
            # Print losses
            if (epoch + 1) % print_every == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                           f'Train Loss: {avg_train_loss:.6f}, '
                           f'Val Loss: {avg_val_loss:.6f}')
        else:
            # Print only training loss if no validation set
            if (epoch + 1) % print_every == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                           f'Train Loss: {avg_train_loss:.6f}')
    
    # Load best model state if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f'Best model loaded with validation loss: {best_val_loss:.6f}')
    
    logging.info('Training completed!')
    return model

# %%
train_indices, temp_indices = train_test_split(np.arange(len(X)), test_size=0.3, random_state=SEED)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.3333, random_state=SEED)

# %%
# [{}, {}, ...*239] -> {[], [], ...*32}

# %%
layers = X[0].keys()

X_dict_temp = {i: [] for i in layers}
Y_dict_temp = {i: [] for i in layers}

for inp, out in zip(X, Y):
    for layer in layers:
        X_dict_temp[layer].append(inp[layer])
        Y_dict_temp[layer].append(out[layer])
    

X_dict = {i: torch.cat(X_dict_temp[i]).to("cuda") for i in X_dict_temp}
Y_dict = {i: torch.cat(Y_dict_temp[i]).to("cuda") for i in Y_dict_temp}

print(X_dict[-1].shape)
print(Y_dict[-1].shape)

# %%
X_train = {layer: X_dict[layer][train_indices] for layer in X_dict}
X_val = {layer: X_dict[layer][val_indices] for layer in X_dict}
X_test = {layer: X_dict[layer][test_indices] for layer in X_dict}

Y_train = {layer: Y_dict[layer][train_indices] for layer in Y_dict}
Y_val = {layer: Y_dict[layer][val_indices] for layer in Y_dict}
Y_test = {layer: Y_dict[layer][test_indices] for layer in Y_dict}

# %%
# Normalize the data
# X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
# Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0) + 1e-8

# X_normalized = (X_train - X_mean) / X_std
# Y_normalized = (Y_train - Y_mean) / Y_std

# %%
X_mean = {layer: X_train[layer].mean(axis=0) for layer in X_train}
X_std = {layer: X_train[layer].std(axis=0) + 1e-8 for layer in X_train}

Y_mean = {layer: Y_train[layer].mean(axis=0) for layer in Y_train}
Y_std = {layer: Y_train[layer].std(axis=0) + 1e-8 for layer in Y_train}

# %%
X_train_normalized = {layer: (X_train[layer] - X_mean[layer]) / X_std[layer] for layer in X_train}
X_val_normalized = {layer: (X_val[layer] - X_mean[layer]) / X_std[layer] for layer in X_val}
X_test_normalized = {layer: (X_test[layer] - X_mean[layer]) / X_std[layer] for layer in X_test}

Y_train_normalized = {layer: (Y_train[layer] - Y_mean[layer]) / Y_std[layer] for layer in Y_train}
Y_val_normalized = {layer: (Y_val[layer] - Y_mean[layer]) / Y_std[layer] for layer in Y_val}
Y_test_normalized = {layer: (Y_test[layer] - Y_mean[layer]) / Y_std[layer] for layer in Y_test}

# %%
print(X_train_normalized[-10].shape)
print(X_val_normalized[-10].shape)
print(X_test_normalized[-10].shape)

print(Y_train_normalized[-10].shape)
print(Y_val_normalized[-10].shape)
print(Y_test_normalized[-10].shape)

# %%
# X_layer = fetch_layer(X, layer) # 239*4096
# Y_layer = fetch_layer(Y, layer) # 239*4096

# # ToDo: Try to normalize to std 1

# X_train_split = X_layer[train_indices]
# Y_train_split = Y_layer[train_indices]

# X_val_split = X_layer[val_indices]
# Y_val_split = Y_layer[val_indices]

# X_test_split = X_layer[test_indices]
# Y_test_split = Y_layer[test_indices]


# train_dataset = TransformDataset(X_train_split, Y_train_split)
# val_dataset = TransformDataset(X_val_split, Y_val_split)

# # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# %%
# num_epochs = 5
num_epochs = 10

lr = 0.001
# lr = 0.01


layers = list(range(-1, -llm.language_model.config.num_hidden_layers, -1))
# all_predictors = {}

for layer in layers:
    print(f"Running for layer {layer}.")

    train_dataset = TransformDataset(X_train_normalized[layer], Y_train_normalized[layer])
    val_dataset = TransformDataset(X_val_normalized[layer], Y_val_normalized[layer])

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    ode_predictor = NeuralODEModel(input_dim=4096, hidden_dim=4096)

    # all_predictors[layer] = train_model(
    trained_ode_predictor = train_model(
        ode_predictor, 
        train_loader, 
        val_loader=val_loader,
        num_epochs=num_epochs, 
        lr=lr,
        print_every=1,
        log_file=f'logs/node_training_try1.log'
    )

    with open(f'ode_ckpt/try1/ode_layer{layer}.pickle', 'wb') as file:
        pickle.dump(trained_ode_predictor, file)
    
    del ode_predictor
    del trained_ode_predictor

    # break


# %%
all_stats = {
    # 'all_predictors': all_predictors,
    'X_mean': X_mean,
    'X_std': X_std,
    'Y_mean': Y_mean,
    'Y_std': Y_std,
}

with open('ode_ckpt/try1/all_stats.pickle', 'wb') as file:
    pickle.dump(all_stats, file)

# %%



