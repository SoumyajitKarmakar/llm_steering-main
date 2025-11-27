import torch
from torch.linalg import solve
import time
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
from tqdm import tqdm
from torcheval.metrics.functional import r2_score

def euclidean_distances_M_2(samples, samples_M_applied, centers, centers_M_applied, threshold=1e-3):
    samples_norm = samples_M_applied * samples
    samples_norm = torch.sum(samples_norm, dim=1)

    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = centers_M_applied * centers
        centers_norm = torch.sum(centers_norm, dim=1)

    samples_norm = torch.reshape(samples_norm, (-1, 1))
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples_M_applied @ centers.T
  
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)

    if samples is centers:
        # print('is centers')
        distances.fill_diagonal_(0)
    else:
        distances = torch.where(abs(distances) < threshold, 0, distances)

    distances.clamp_(min=0)
    distances.sqrt_()

    return distances

def laplacian_M_2(bandwidth, distances):
    assert bandwidth > 0
    kernel_mat = distances
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def laplacian_M_3(X, X_M_applied, x, x_M_applied, L):
    dist = euclidean_distances_M_2(X, X_M_applied, x, x_M_applied)


    K = laplacian_M_2(L, dist.clone())
    K = K/dist

    K[K == float("Inf")] = 0.

    return K


def laplacian_M(X, X_M_applied, x, x_M_applied, L):
    dist = euclidean_distances_M_2(X, X_M_applied, x, x_M_applied)
    K = laplacian_M_2(L, dist.clone())

    K[K == float("Inf")] = 0.

    return K


def get_grads_2(X, x, sol, L, P):
    """
    Computes the Average Gradient Outer Product (AGOP).
    Handles multi-variate outputs by summing the AGOPs for each output dimension.
    
    Args:
        X: Training inputs (n, d)
        x: Evaluation inputs (m, d)
        sol: Dual coefficients (c, n) or (1, n)
        L: Bandwidth
        P: Current transformation matrix
    """

    X_M_applied = X @ P
    x_M_applied = X_M_applied if x is X else x @ P

    K = laplacian_M_3(X, X_M_applied, x, x_M_applied, L) # (n, m)

    n, d = X.shape
    m, d = x.shape
    
    # Check if sol is multi-variate
    # sol is expected to be (c, n) coming from solve_kr (transposed)
    if sol.dim() == 1:
        sol = sol.unsqueeze(0) # (1, n)
    c, n_sol = sol.shape
    assert n_sol == n

    # step_1
    # X1: (n, d)
    X1 = X_M_applied
    
    # We want to compute sum_j (alpha_j * X1) for each output j
    # sol: (c, n)
    # X1: (n, d)
    # We need separate gradients for each c.
    # sol_X1: (n, c, d)
    
    # sol.T is (n, c). 
    # sol.T.unsqueeze(2) is (n, c, 1)
    # X1.unsqueeze(1) is (n, 1, d)
    sol_X1 = sol.T.unsqueeze(2) * X1.unsqueeze(1) # (n, c, d)

    # K.T is (m, n)
    # step_1 = K.T @ sol_X1
    # (m, n) @ (n, c, d) -> (m, c, d)
    step_1 = torch.einsum('mn,ncd->mcd', K.T, sol_X1)

    # step_2
    # sol_K = sol @ K
    # (c, n) @ (n, m) -> (c, m)
    sol_K = sol @ K 
    
    # sol_K.T is (m, c)
    # sol_K.T.unsqueeze(2) is (m, c, 1)
    # x1 = x_M_applied: (m, d)
    # x1.unsqueeze(1) is (m, 1, d)
    x1 = x_M_applied
    step_2 = sol_K.T.unsqueeze(2) * x1.unsqueeze(1) # (m, c, d)
    
    G = (step_1 - step_2) # (m, c, d)
    
    # Center gradients? Original code: G = G - torch.mean(G, axis=0, keepdims=True)
    # This centers across samples 'm'.
    G = G - torch.mean(G, dim=0, keepdim=True)

    # Compute M = sum_c G_c^T G_c
    # G is (m, c, d). We want to sum over m and c.
    # M = einsum('mcd,mce->de', G, G)
    M = torch.einsum('mcd,mce->de', G, G)  # (d, d)
    
    # Normalize
    M = M / (m * c * L * L) # Average over m and c? Original was m*L*L. 
                            # If we sum over c, the scale increases. 
                            # Keeping it proportional to original logic.

    # import pdb; pdb.set_trace()


    return M #/ M.max()

def solve_kr(X, X_M_applied, y, L, reg):
    K = laplacian_M(X, X_M_applied, X, X_M_applied, L)    

    # y can be (n, c)
    try: 
        # solve(A, B) solves AX = B. 
        # K is (n, n). y is (n, c).
        # Returns (n, c).
        # We transpose to return (c, n) to match original signature expectation
        sol = solve(K + reg * torch.eye(len(K), device=X.device), y).T    
    except torch._C._LinAlgError:
        return None
    return sol

def get_err(sol, X, x, X_M_applied, x_M_applied, y, L):
    # sol: (c, n)
    # y: (m, c)
    K_test = laplacian_M(X, X_M_applied, x, x_M_applied, L) # (n, m)
    
    # preds = (sol @ K_test).T
    # (c, n) @ (n, m) -> (c, m). T -> (m, c)
    preds = (sol @ K_test).T
    
    mse = torch.mean(torch.square(preds - y))
    
    # r2_score handles multi-output? 
    # torcheval r2_score expects (preds, target).
    # If multi-dimensional, it might average or return raw.
    # We'll assume we want the average R2 across dimensions.
    r2 = r2_score(preds, y, multioutput='uniform_average')
    
    return mse, r2


def get_top_dir_err(X, y, M):
    epsilon = 1e-6  # Small regularization factor # orig
    # epsilon = 1e-3  # qwen test
    max_attempts = 3  # Number of times to try increasing regularization

    # start = time.time()
    for attempt in range(max_attempts):
        try:
            # S, U = torch.linalg.eigh(concept_features)
            s, u = torch.lobpcg(M, k=1)
            break  # If successful, exit the loop
        except torch._C._LinAlgError:
            epsilon *= 10  # Increase regularization
            print(f"Warning: Matrix ill-conditioned. Retrying with epsilon={epsilon}")
            # concept_features += epsilon * torch.eye(M.shape[0], device=M.device) # orig
            M += epsilon * torch.eye(M.shape[0], device=M.device)
    else:
        # raise RuntimeError("linalg.eigh failed to converge even with regularization.") # orig
        print("linalg.eigh failed to converge even with regularization.")
        return None, None

    # s, u = torch.lobpcg(M, k=1)
    preds = X @ u

    # print(preds, y)
    # print(preds.shape, y.shape)
    
    # If y is multi-variate, correlation with 1D projection is not well-defined 
    # without specifying which dimension.
    # We return the average absolute correlation with the first principal component of y as a proxy,
    # or just return 0 if we rely on get_err for selection.
    if y.shape[1] > 1:
        # Fallback: just return 0 correlation, rely on R2 from get_err for selection
        return 0.0, u
        
    return torch.abs(torch.corrcoef(torch.cat((preds, y), dim=-1).T))[0, 1].item(), u

def rfm(traindata, testdata, L=10, reg=1e-3, num_iters=10, norm=False):
    X_train, y_train = traindata
    X_test, y_test = testdata

    mean = torch.mean(X_train, dim=0, keepdims=True)
    X_train = (X_train - mean)
    X_test = (X_test - mean)

    if norm:
        X_train = X_train / torch.norm(X_train, dim=-1).reshape(-1, 1)
        X_test = X_test / torch.norm(X_test, dim=-1).reshape(-1, 1)

    n, d = X_train.shape
    M = torch.eye(d, device=X_train.device)

    best_r = -float('inf')
    best_r2 = -float('inf')
    best_u = None
    
    # Check if multi-variate
    is_multivariate = y_train.shape[1] > 1

    for i in range(num_iters):
        X_train_M_applied = X_train @ M
        X_test_M_applied = X_test @ M # Needed for get_err
        
        sol = solve_kr(X_train, X_train_M_applied, y_train, L, reg)
        # print(sol)
        if sol is None: 
            break
        
        if is_multivariate:
            # Use R2 for selection in multi-variate case
            test_mse, test_r2 = get_err(sol, X_train, X_test, X_train_M_applied, X_test_M_applied, y_test, L)
            
            # Compute u anyway as it's the steering vector
            _, u = get_top_dir_err(X_test, y_test, M) # Ignore correlation return
            
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_u = u.clone() if u is not None else None
                
        else:
            # Use correlation of top direction for scalar case (original behavior)
            test_r, u = get_top_dir_err(X_test, y_test, M)
            
            if u is None:
                continue

            if test_r > best_r: 
                best_r = test_r
                best_u = u.clone()

        M = get_grads_2(X_train, X_train, sol, L, M)
        M /= M.max() # orig, disabled for qwen tests

    if is_multivariate:
        return best_u, best_r2
    else:
        return best_u, best_r 

def main():

    # create low rank data
    n = 1000
    d = 100
    torch.manual_seed(0)
    X_train = torch.randn(n,d).cuda()
    X_test = torch.randn(n,d).cuda()

    # y_train = torch.where(X_train[:, 0] > 0, 1., 0).reshape(-1, 1)
    # y_test = torch.where(X_test[:, 0] > 0, 1., 0).reshape(-1, 1)

    # Multi-variate example
    # y1 = x0 + x1
    # y2 = x0 - x1
    y_train_1 = ((X_train[:, 0] + X_train[:, 1])).reshape(-1, 1)
    y_train_2 = ((X_train[:, 0] - X_train[:, 1])).reshape(-1, 1)
    y_train = torch.cat([y_train_1, y_train_2], dim=1)

    y_test_1 = ((X_test[:, 0] + X_test[:, 1])).reshape(-1, 1)
    y_test_2 = ((X_test[:, 0] - X_test[:, 1])).reshape(-1, 1)
    y_test = torch.cat([y_test_1, y_test_2], dim=1)

    print(X_train.shape, y_train.shape)

    start = time.time()
    best_u, best_metric = rfm((X_train, y_train), 
                 (X_test, y_test),
                 reg=1e-3,
                 L=10,
                 num_iters=10)
    
    print("Best U shape:", best_u.shape if best_u is not None else "None")
    print("Best Metric (R2 for multi, r for scalar):", best_metric)

    end = time.time()
    print("Training time: ", end - start)    

if __name__ == "__main__":
    main()
