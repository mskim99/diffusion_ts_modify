import torch
import torch.nn.functional as F
import numpy as np

from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from sklearn.metrics.pairwise import pairwise_kernels

import lpips
from scipy.stats import entropy

def standardize(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    return (data - mean) / std

# A: covariance matrix (positive semi-definite)
def matrix_sqrt_newton_schulz(A, num_iters=50):
    batch = A.shape[0]
    dim = A.shape[1]
    normA = A.norm()
    Y = A / normA
    I = torch.eye(dim, device=A.device).expand(batch, -1, -1)
    Z = torch.eye(dim, device=A.device).expand(batch, -1, -1)

    for i in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    return Y * torch.sqrt(normA)

def rmse(predictions, targets):
    # Compute the differences for the last dimension
    differences = predictions[:, 0, :] - targets[:, 0, :]

    # Square the differences and compute the mean along the last dimension, then take the square root
    if isinstance(differences, np.ndarray):
        rmse_values = np.sqrt((differences ** 2).mean(axis=-1))
        return rmse_values
    elif isinstance(differences, torch.Tensor):
        rmse_values = torch.sqrt((differences ** 2).mean(axis=-1))
        return rmse_values
    else:
        raise ValueError("Both inputs should be either numpy arrays or torch tensors")

def psnr(img1, img2, max_val=1.0):
    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()

    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img1 = 2. * img1 - 1.
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    img2 = 2. * img2 - 1.

    mse = F.mse_loss(img1, img2, reduction='none').mean([1, 2])

    psnr_val = 10 * torch.log10(max_val ** 2 / mse)

    if isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        return psnr_val.numpy()

    return psnr_val

def cosine_sim(v1, v2):
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
    elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        dot_product = torch.dot(v1, v2)
        norm_v1 = torch.linalg.norm(v1)
        norm_v2 = torch.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
    else:
        raise ValueError("Both inputs should be either numpy arrays or torch tensors")

def fre_cosine(signal1, signal2):
    # check numpy or tensor
    if isinstance(signal1, np.ndarray) and isinstance(signal2, np.ndarray):
        spectrum1 = np.fft.fft(signal1, axis=-1)
        spectrum2 = np.fft.fft(signal2, axis=-1)
        cos_sim_func = cosine_sim
    elif isinstance(signal1, torch.Tensor) and isinstance(signal2, torch.Tensor):
        signal1 = signal1.cpu()
        signal2 = signal2.cpu()

        spectrum1 = torch.fft.fft(signal1, dim=-1)
        spectrum2 = torch.fft.fft(signal2, dim=-1)
        cos_sim_func = torch.nn.functional.cosine_similarity
    else:
        raise ValueError("Both inputs should be either numpy arrays or torch tensors")

    # calculate similarity
    B = signal1.shape[0]
    similarities = np.zeros(B) if isinstance(signal1, np.ndarray) else torch.zeros(B)
    for i in range(B):
        if isinstance(signal1, np.ndarray):
            similarities[i] = cos_sim_func(np.abs(spectrum1[i, 0, :]), np.abs(spectrum2[i, 0, :]))
        else:
            similarities[i] = cos_sim_func(torch.abs(spectrum1[i, :]), torch.abs(spectrum2[i, :]))

    return similarities

def compute_ssim_1d(x, y):
    # skimage ssim은 2D 이상만 허용하므로 reshape
    return ssim(x, y, data_range=y.max() - y.min())

def compute_lpips_1d(x, y, net='alex'):
    # LPIPS는 [1, 3, H, W] 형태의 텐서를 입력으로 받음
    x_tensor = torch.tensor(x).view(1, 1, 1, -1).repeat(1, 3, 1, 1)  # [1, 3, 1, N]
    y_tensor = torch.tensor(y).view(1, 1, 1, -1).repeat(1, 3, 1, 1)
    loss_fn = lpips.LPIPS(net=net)
    return loss_fn(x_tensor, y_tensor).item()

def eval_fid(real_data, gen_data):
    # 평균과 공분산 계산
    real_d = real_data.reshape(real_data.shape[0], -1)
    gen_d = gen_data.reshape(gen_data.shape[0], -1)

    mu1, sigma1 = real_d.mean(axis=0), np.cov(real_d, rowvar=False)
    mu2, sigma2 = gen_d.mean(axis=0), np.cov(gen_d, rowvar=False)

    # FID 계산 (Fréchet Distance)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):  # 수치적 허용
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def eval_fid_torch(real_data, gen_data, device='cuda'):
    # reshape and move to device
    real_d = real_data.view(real_data.size(0), -1).to(device)
    gen_d = gen_data.view(gen_data.size(0), -1).to(device)

    # mean and covariance
    mu1 = real_d.mean(dim=0)
    mu2 = gen_d.mean(dim=0)

    sigma1 = torch.cov(real_d.T)
    sigma2 = torch.cov(gen_d.T)

    # Move to CPU for sqrtm (scipy 사용을 위함)
    '''
    mu1_np = mu1.cpu().numpy()
    mu2_np = mu2.cpu().numpy()
    sigma1_np = sigma1.cpu().numpy()
    sigma2_np = sigma2.cpu().numpy()
    '''

    # diff = mu1_np - mu2_np
    # covmean, _ = sqrtm(sigma1_np @ sigma2_np, disp=False)
    diff = mu1 - mu2
    covmean, _ = matrix_sqrt_newton_schulz(sigma1 @ sigma2)
    if torch.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def mmd(real_data, gen_data, kernel='rbf', gamma=None):
    """Maximum Mean Discrepancy with kernel (default: RBF)"""
    X = real_data
    Y = gen_data
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

def emd(real_data, gen_data):
    """Earth Mover's Distance for 1D signals: 평균 EMD over samples"""
    emd_total = 0.0
    real_d = real_data.reshape(1, -1)
    gen_d = gen_data.reshape(1, -1)
    for r, g in zip(real_d, gen_d):
        emd_total += wasserstein_distance(r, g)
    return emd_total / len(real_data)

def eval_is(preds, splits=10):
    """
    preds: np.array of shape (N, num_classes) – softmaxed class probabilities
    splits: number of splits (default 10)

    Returns:
        mean IS, std IS
    """
    N = preds.shape[0]
    split_scores = []

    for k in range(splits):
        part = preds[k * N // splits: (k + 1) * N // splits]
        py = np.mean(part, axis=0)
        scores = [entropy(pyx, py) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def eval_all(signal1, signal2):
    rmse_all=rmse(signal1, signal2)
    psnr_all=psnr(signal1, signal2)
    fre_cos_all=fre_cosine(signal1, signal2)
    # ssim_all = compute_ssim_1d(signal1, signal2)
    # lpips_all = compute_lpips_1d(signal1, signal2)
    # fid_all = fid(signal1, signal2)
    mmd_all = mmd(signal1, signal2)
    emd_all = emd(signal1, signal2)
    return [rmse_all,psnr_all,fre_cos_all, mmd_all, emd_all]
