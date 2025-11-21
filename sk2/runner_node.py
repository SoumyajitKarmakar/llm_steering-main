import sys
import os
from pathlib import Path

notebook_path = "/u/skarmakar1/version_check/llm_steering-main/sk"
sys.path.append(notebook_path)

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


# %%


# %% [markdown]
# Neural ODE

# %%
with open("../data/moods/all_antonym_pairs.pkl", 'rb') as file:
    all_e = pickle.load(file)

# %%
print("Total data:", len(all_e))
print(all_e[:5])

train_data_t, test_data = train_test_split(all_e, test_size=0.1, random_state=SEED)

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

# %%
print(test_data)

# %%
X_train, Y_train = read_tuples(llm, train_data, path='../directions_moods_plus_llama/')

# %%
print(X_train[-1].shape)
print(Y_train[-1].shape)

# %%
X_test, Y_test = read_tuples(llm, test_data, path='../directions_moods_plus_llama/')

# %%
print(X_test[-1].shape)
print(Y_test[-1].shape)

# %%
X_mean = {layer: X_train[layer].mean(axis=0, keepdim=True) for layer in X_train}
X_std = {layer: X_train[layer].std(axis=0, keepdim=True) + 1e-8 for layer in X_train}

Y_mean = {layer: Y_train[layer].mean(axis=0, keepdim=True) for layer in Y_train}
Y_std = {layer: Y_train[layer].std(axis=0, keepdim=True) + 1e-8 for layer in Y_train}

# %%
print(X_mean[-1].shape)
print(X_std[-1].shape)

# %%
X_train_normalized = {layer: (X_train[layer] - X_mean[layer]) / X_std[layer] for layer in X_train}
X_test_normalized = {layer: (X_test[layer] - X_mean[layer]) / X_std[layer] for layer in X_test}

Y_train_normalized = {layer: (Y_train[layer] - Y_mean[layer]) / Y_std[layer] for layer in Y_train}
Y_test_normalized = {layer: (Y_test[layer] - Y_mean[layer]) / Y_std[layer] for layer in Y_test}

# %%
print(X_train_normalized[-1])
print(X_train_normalized[-1].shape)
print(Y_train_normalized[-1].shape)

# %%


# %%
# pip install torchdiffeq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint as odeint  # O(1)-memory adjoint

from copy import deepcopy
import logging

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
# 1) Neural vector field f_theta(t, x): R^4096 -> R^4096, implements dx/dt
class VectorField(nn.Module):
    def __init__(self, dim=4096, hidden=1024, use_time=True):
        super().__init__()
        in_dim = dim + (1 if use_time else 0)
        self.use_time = use_time
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, t, x):
        if self.use_time:
            t_feat = torch.full((x.shape[0], 1), float(t), device=x.device, dtype=x.dtype)
            z = torch.cat([x, t_feat], dim=1)
        else:
            z = x
        dx = self.net(z)
        return self.norm(dx)

# 2) ODE block that integrates from t0=0 to t1=1
class NeuralODE(nn.Module):
    def __init__(self, vf, method='dopri5', rtol=1e-5, atol=1e-5):
        super().__init__()
        self.vf = vf
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, y0):
        t = torch.tensor([0.0, 1.0], device=y0.device, dtype=y0.dtype)
        y = odeint(self.vf, y0, t, method=self.method, rtol=self.rtol, atol=self.atol)
        return y[-1]

# 3) Training with optional validation; prints train and validation loss each epoch
def train_neural_ode(
    X, Y, X_val=None, Y_val=None, *,
    epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-3, hidden=1024, device=None, log_file="ode_logs/training.log",
):

    setup_logger(log_file)

    assert X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape and X.shape[1] == 4096
    if X_val is not None and Y_val is not None:
        assert X_val.ndim == 2 and Y_val.ndim == 2 and X_val.shape == Y_val.shape and X_val.shape[1] == 4096
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("*"*50)
    logging.info(f'Starting training on device: {device}')
    logging.info(f'Number of epochs: {epochs}, Learning rate: {lr}')

    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_dl = None
    if X_val is not None and Y_val is not None:
        val_ds = TensorDataset(X_val, Y_val)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    vf = VectorField(dim=4096, hidden=hidden, use_time=True).to(device)
    model = NeuralODE(vf, method='dopri5', rtol=1e-5, atol=1e-5).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(ds)

        # Validate (minimal additions)
        if val_dl is not None:
            model.eval()  # set eval mode for layers like dropout/BN
            vrunning = 0.0
            with torch.no_grad():  # disable autograd for speed/memory
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    vpred = model(xb)
                    vloss = loss_fn(vpred, yb)
                    vrunning += vloss.item() * xb.size(0)
            val_loss = vrunning / len(val_dl.dataset)
            # print(f"epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
            logging.info(f"epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            # Save best model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())

        else:
            # print(f"epoch {epoch}: train_loss={train_loss:.6f}")
            logging.info(f"epoch {epoch}: train_loss={train_loss:.6f}")

    # Load best model state if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f'Best model loaded with validation loss: {best_val_loss:.6f}')
    
    return model

# # 4) Example usage (replace with your actual tensors)
# if __name__ == "__main__":
#     N, Nv = 512, 128
#     X = torch.randn(N, 4096)
#     W = torch.randn(4096, 4096) / 4096**0.5
#     Y = X @ W.T

#     X_val = torch.randn(Nv, 4096)
#     Y_val = X_val @ W.T

#     model = train_neural_ode(X, Y, X_val=X_val, Y_val=Y_val,
#                              epochs=10, batch_size=16, lr=1e-3, hidden=1024)

# %%


# %%


# %%
# all_node = {}

for layer in X_train_normalized:
    print(f"Layer: {layer}")
    X = X_train_normalized[layer]
    Y = Y_train_normalized[layer]

    X_val = X_test_normalized[layer]
    Y_val = Y_test_normalized[layer]

    # model = train_neural_ode(X, Y, X_val=X_val, Y_val=Y_val, epochs=10, batch_size=16, lr=1e-3, hidden=1024)
    # model = train_neural_ode(X, Y, X_val=X_val, Y_val=Y_val, epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-3, hidden=256, log_file="ode_logs/training_try1.log") # decent
    model = train_neural_ode(X, Y, X_val=X_val, Y_val=Y_val, epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-4, hidden=256, log_file="ode_logs/training_try1.log")

    # all_node[layer] = model

    with open(f'ode_ckpt/try1/ode_layer{layer}.pickle', 'wb') as file:
        pickle.dump(model, file)
    
    del model

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
