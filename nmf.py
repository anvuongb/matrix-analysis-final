import numpy as np
import numpy.linalg
import time
from IPython.display import clear_output

def accelerated_MU(X, L, alpha, epsillon, max_iter, save_every_iter=-1, save_with_index=False, save_cb=None, weight_path="", restart=False):
    """
    X: input matrix X (MxN)
    L: low rank, decompose X=WH where W (MxL) and H (LxN)
    alpha: control parameter for inner loop
    epsillon: control paramter for stopping condition in inner loop
    max_iter: number of iterations for algorithm
    save_every_iter: save weight every # iterations
    save_cb: save function callback
    weight_path
    """
    print(f"Start MU with L={L}, alpha={alpha}, epsillon={epsillon}, max_iter={max_iter}")
    P = X.shape[0]*X.shape[1]
    M = X.shape[0]
    N = X.shape[1]
    
    W = np.random.rand(M, L)
    H = np.random.rand(L, N)
    
    ro_W = 1 + (P + N*L)/(M*L + M)
    ro_H = 1 + (P + M*L)/(N*L + N)
    
    print(f"P={P}, M={M}, N={N}, L={L}")
    print(f"ro_W={ro_W}, ro_H={ro_H}")
    
    norm_error = []
    e_time = []
    start_idx = 0

    if restart:
        W = np.load(weight_path + "/weights/matrix_W.npy")
        H = np.load(weight_path + "/weights/matrix_H.npy")
        norm_error = list(np.load(weight_path + "/weights/norm_error.npy"))
        e_time = list(np.load(weight_path + "/weights/e_time.npy"))
        start_idx = len(e_time)
        print(f"restarting from {weight_path} at idx {start_idx}")

    start = time.time()
    for i in np.arange(start_idx, max_iter):
        # update W
        A = np.matmul(X, np.transpose(H))
        B = np.matmul(H, np.transpose(H))
        W_norm_0 = 0
        W_norm_current = 0
        for j in np.arange(int(1+alpha*ro_W)):
            W_last = W
            C = np.matmul(W_last, B)
            W = np.multiply(W_last, np.divide(A, C))
            if j==0:
                W_norm_0 = np.linalg.norm(W-W_last, 'fro')**2
            W_norm_current = np.linalg.norm(W-W_last, 'fro')**2
            if W_norm_current<=epsillon*W_norm_0:
                break
        
        # update H
        A = np.matmul(np.transpose(W), X)
        B = np.matmul(np.transpose(W), W)
        H_norm_0 = 0
        H_norm_current = 0
        for j in np.arange(int(1+alpha*ro_H)):
            H_last = H
            C = np.matmul(B, H)
            H = np.multiply(H, np.divide(A, C))
            if j==0:
                H_norm_0 = np.linalg.norm(H-H_last, 'fro')**2
            H_norm_current = np.linalg.norm(H-H_last, 'fro')**2
            if H_norm_current<=epsillon*H_norm_0:
                break
        
        elapsed_time = time.time()-start
        current_norm_error = np.linalg.norm(X-np.matmul(W, H), 'fro')
        norm_error.append(current_norm_error)
        e_time.append(elapsed_time)
        print(f"{i}, elapsed time {time.time()-start}, current err {current_norm_error}")
        if i%20 == 0:
            clear_output(wait=True)
        if save_every_iter > 0:
            if (i+1) % save_every_iter == 0:
                if save_with_index:
                    save_cb(W, H, norm_error, e_time, weight_path, i+1)
                else:
                    save_cb(W, H, norm_error, e_time, weight_path, -1)
    return W, H, norm_error, e_time 

def accelerated_HALS(X, L, alpha, epsillon, max_iter, update_func="gillis", save_every_iter=-1, save_with_index=False, save_cb=None, weight_path="", restart=False):
    """
    X: input matrix X (MxN)
    L: low rank, decompose X=WH where W (MxL) and H (LxN)
    alpha: control parameter for inner loop
    epsillon: control paramter for stopping condition in inner loop
    max_iter: number of iterations for algorithm
    update_func: select update function, 
                 "paper" uses the procedure shown in paper - Algorithm 4-5, 
                 "gillis" adapts the procedure from Gillis Matlab, with sligh modifications in indices
    save_every_iter: save weight every # iterations
    save_cb: save function callback
    weight_path
    """
    print(f"Start HALS with L={L}, alpha={alpha}, epsillon={epsillon}, max_iter={max_iter}, update_func={update_func}")
    def update_once(F, A, B, L):
        indices = range(L)
        for k in indices:
            C = (A[[k],:]-np.matmul(B[[k],:][:,indices], F[indices,:]))/B[k,k]
            F[[k],:] = np.maximum(C, 1e-16)
        return F
    
    def update_once_gillis(F, A, B, L):
        indices = range(L)
        for k in indices:
            C = (A[[k],:]-np.matmul(B[[k],:][:,indices], F[indices,:]))/B[k,k]
            F[[k],:] = F[[k],:] + np.maximum(C, -F[[k],:])
            if np.array_equal(F[[k],:], np.zeros((1, np.max(F.shape)))):
                F[[k],:] = 1e-16*np.max(F)
        return F
    
    if update_func == "gillis":
        update = update_once_gillis
    else:
        update = update_once
    
    P = X.shape[0]*X.shape[1]
    M = X.shape[0]
    N = X.shape[1]
    
    W = np.random.rand(M, L)
    H = np.random.rand(L, N)
    
    ro_W = 1 + (P + N*L)/(M*L + M)
    ro_H = 1 + (P + M*L)/(N*L + N)
    
    print(f"P={P}, M={M}, N={N}, L={L}")
    print(f"ro_W={ro_W}, ro_H={ro_H}")
    
    norm_error = []
    e_time = []

    start = time.time()
    
    A = np.matmul(X, np.transpose(H))
    B = np.matmul(H, np.transpose(H))
    # scale based on https://arxiv.org/abs/0810.4225
    scaling = np.sum(np.multiply(A, W))/np.sum(np.multiply(B,np.matmul(np.transpose(W),W)))
    print("scaling=",scaling)
    W = W*scaling

    start_idx = 0

    if restart:
        W = np.load(weight_path + "/weights/matrix_W.npy")
        H = np.load(weight_path + "/weights/matrix_H.npy")
        norm_error = list(np.load(weight_path + "/weights/norm_error.npy"))
        e_time = list(np.load(weight_path + "/weights/e_time.npy"))
        start_idx = len(e_time)
        print(f"restarting from {weight_path} at idx {start_idx}")

    for i in np.arange(start_idx, max_iter):
        if i != 0:
            A = np.matmul(X, np.transpose(H))
            B = np.matmul(H, np.transpose(H))
        # update W                                    
        W_norm_0 = 0
        W_norm_current = 0
        for j in np.arange(int(1+alpha*ro_W)):
            W_last = W.copy()
            W = update(W.T, A.T, B.T, L).T
            if j==0:
                W_norm_0 = np.linalg.norm(W-W_last, 'fro')**2
            W_norm_current = np.linalg.norm(W-W_last, 'fro')**2
            if W_norm_current<=epsillon*W_norm_0:
                break
        
        # update H
        A = np.matmul(np.transpose(W), X)
        B = np.matmul(np.transpose(W), W)
        H_norm_0 = 0
        H_norm_current = 0
        for j in np.arange(int(1+alpha*ro_H)):
            H_last = H.copy()
            H = update(H, A, B, L)
            if j==0:
                H_norm_0 = np.linalg.norm(H-H_last, 'fro')**2
            H_norm_current = np.linalg.norm(H-H_last, 'fro')**2
            if H_norm_current<=epsillon*H_norm_0:
                break
        
        elapsed_time = time.time()-start
        current_norm_error = np.linalg.norm(X-np.matmul(W, H), 'fro')
        norm_error.append(current_norm_error)
        e_time.append(elapsed_time)
        print(f"{i}, elapsed time {time.time()-start}, current err {current_norm_error}")
        if i%20 == 0:
            clear_output(wait=True)
        if save_every_iter > 0:
            if save_with_index:
                    save_cb(W, H, norm_error, e_time, weight_path, i+1)
            else:
                save_cb(W, H, norm_error, e_time, weight_path, -1)
    return W, H, norm_error, e_time 