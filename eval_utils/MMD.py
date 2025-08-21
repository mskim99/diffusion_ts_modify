import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import numpy as np




def cross_correlation_distribution(data):
    def get_lower_triangular_indices_no_diag(n):
        indices = torch.tril_indices(n, n).to(data).long()
        indices_without_diagonal = (indices[0] != indices[1]).nonzero(as_tuple=True)
        return indices[0][indices_without_diagonal], indices[1][indices_without_diagonal]
        
    index = get_lower_triangular_indices_no_diag(data.shape[2])
    toreturn = []
    for i in range(data.shape[0]):
        corr_matrix = torch.corrcoef(data[i].T)
        toreturn.append(corr_matrix[index])
    toreturn = torch.stack(toreturn, dim=0).to(data)
    
    # Replace inf and NaN values with 0
    toreturn = torch.where(torch.isinf(toreturn) | torch.isnan(toreturn), torch.tensor(0.0), toreturn)
    return toreturn.float()


 # From https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(x),
                  torch.zeros(xx.shape).to(x),
                  torch.zeros(xx.shape).to(x))

    if kernel == "multiscale":
        bandwidth_range = [0.01,0.05,0.1,0.2,0.5]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [0.01,0.05,0.1,0.2,0.5,0.7,1.0,1.5,2.0]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)

import torch
from typing import Iterable, Optional

@torch.no_grad()
def mmd_blockwise(
    x: torch.Tensor,              # (N, D)
    y: torch.Tensor,              # (M, D)
    kernel: str = "rbf",          # "rbf" | "multiscale"
    bandwidths: Optional[Iterable[float]] = None,
    chunk_size: int = 1024,
    use_autocast: bool = False,
    force_cpu: bool = False,
) -> torch.Tensor:
    """
    메모리 친화적 MMD^2 (self-term은 대각 제외 평균, cross-term은 전체 평균).
    N×N 전체 행렬을 만들지 않고 블록으로 누적합니다.
    """
    assert kernel in {"rbf", "multiscale"}
    if bandwidths is None:
        bandwidths = ([0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0]
                      if kernel == "rbf" else [0.01, 0.05, 0.1, 0.2, 0.5])

    dev = torch.device("cpu") if force_cpu else (x.device if x.is_cuda else torch.device("cpu"))
    x = x.to(dev, non_blocking=True)
    y = y.to(dev, non_blocking=True)

    N, M = x.size(0), y.size(0)
    sum_kxx = torch.zeros((), dtype=torch.float64, device=dev)
    sum_kyy = torch.zeros((), dtype=torch.float64, device=dev)
    sum_kxy = torch.zeros((), dtype=torch.float64, device=dev)
    cnt_xx = 0
    cnt_yy = 0
    cnt_xy = 0

    def kernel_from_sqdist(d2: torch.Tensor) -> torch.Tensor:
        if kernel == "rbf":
            out = 0.0
            for a in bandwidths:
                out = out + torch.exp(-0.5 * d2 / float(a))
            return out
        else:  # multiscale
            out = 0.0
            for a in bandwidths:
                a2 = float(a) ** 2
                out = out + (a2 / (a2 + d2))
            return out

    autocast_ctx = (
        torch.cuda.amp.autocast if (use_autocast and dev.type == "cuda")
        else torch.cpu.amp.autocast
    )

    with autocast_ctx(enabled=use_autocast):
        # Kxx (대각 제외)
        for i in range(0, N, chunk_size):
            xi = x[i:i+chunk_size]
            b1 = xi.size(0)
            for j in range(i, N, chunk_size):
                xj = x[j:j+chunk_size]
                d2 = torch.cdist(xi, xj, p=2) ** 2       # (b1, b2)
                kij = kernel_from_sqdist(d2).to(torch.float64)
                if i == j:
                    mask = torch.triu(torch.ones(b1, b1, dtype=torch.bool, device=dev), diagonal=1)
                    sum_kxx += kij[mask].sum()
                    cnt_xx  += int(mask.sum().item())
                else:
                    sum_kxx += kij.sum()
                    cnt_xx  += (b1 * xj.size(0))

        # Kyy (대각 제외)
        for i in range(0, M, chunk_size):
            yi = y[i:i+chunk_size]
            b1 = yi.size(0)
            for j in range(i, M, chunk_size):
                yj = y[j:j+chunk_size]
                d2 = torch.cdist(yi, yj, p=2) ** 2
                kij = kernel_from_sqdist(d2).to(torch.float64)
                if i == j:
                    mask = torch.triu(torch.ones(b1, b1, dtype=torch.bool, device=dev), diagonal=1)
                    sum_kyy += kij[mask].sum()
                    cnt_yy  += int(mask.sum().item())
                else:
                    sum_kyy += kij.sum()
                    cnt_yy  += (b1 * yj.size(0))

        # Kxy (전체)
        for i in range(0, N, chunk_size):
            xi = x[i:i+chunk_size]
            b1 = xi.size(0)
            for j in range(0, M, chunk_size):
                yj = y[j:j+chunk_size]
                d2 = torch.cdist(xi, yj, p=2) ** 2
                kij = kernel_from_sqdist(d2).to(torch.float64)
                sum_kxy += kij.sum()
                cnt_xy  += (b1 * yj.size(0))

    kxx = sum_kxx / max(cnt_xx, 1)
    kyy = sum_kyy / max(cnt_yy, 1)
    kxy = sum_kxy / max(cnt_xy, 1)
    return (kxx + kyy - 2.0 * kxy).to(x.dtype)


@torch.no_grad()
def BMMD_memory_friendly(
    x: torch.Tensor,              # (N, C) 또는 (N, C, D_feat) -> 각 채널별 (N, D) 처리
    y: torch.Tensor,
    kernel: str = "rbf",
    bandwidths: Optional[Iterable[float]] = None,
    channel_dim: int = 1,
    chunk_size: int = 1024,
    use_autocast: bool = True,
    force_cpu: bool = False,
) -> torch.Tensor:
    """
    채널별 MMD를 안전하게 계산. 루프 안에서 .to(device)와 empty_cache() 금지.
    """
    # 채널 차원을 앞으로
    if channel_dim != 1:
        x = x.movedim(channel_dim, 1)
        y = y.movedim(channel_dim, 1)

    dev = torch.device("cpu") if force_cpu else (x.device if x.is_cuda else torch.device("cpu"))
    x = x.to(dev, non_blocking=True)
    y = y.to(dev, non_blocking=True)

    C = x.size(1)
    out = []
    for c in range(C):
        xc = x[:, c]
        yc = y[:, c]
        # (N,)이면 (N,1)로 맞춰 2D 보장
        if xc.ndim == 1: xc = xc.unsqueeze(-1)
        if yc.ndim == 1: yc = yc.unsqueeze(-1)
        mmd2 = mmd_blockwise(
            xc, yc, kernel=kernel, bandwidths=bandwidths,
            chunk_size=chunk_size, use_autocast=use_autocast, force_cpu=force_cpu
        )
        out.append(mmd2)
    return torch.stack(out).to(x.dtype)   # (C,)


def BMMD(x, y, kernel):
    """Empirical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P : Shape (Feature, Sample, 1)
        y: second sample, distribution Q : Shape (Feature, Sample, 1)
        kernel: kernel type such as "multiscale" or "rbf"
    """
    # Compute matrix products
    xx = torch.bmm(x, x.transpose(1, 2))
    yy = torch.bmm(y, y.transpose(1, 2))
    zz = torch.bmm(x, y.transpose(1, 2))

    # Compute diagonal matrices
    rx = torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(2).expand_as(xx).transpose(1,2)
    ry = torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(2).expand_as(yy).transpose(1,2)


    # Compute squared Euclidean distances
    dxx = rx.transpose(1, 2) + rx - 2. * xx
    dyy = ry.transpose(1, 2) + ry - 2. * yy
    dxy = rx.transpose(1, 2) + ry - 2. * zz
    
    # Initialize tensors for results
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(xx)
    XY = torch.zeros_like(xx)

    if kernel == "multiscale":
        bandwidth_range = [0.01, 0.05, 0.1, 0.2, 0.5]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [0.01,0.05,0.1,0.2,0.5,0.7,1.0]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY, [1,2])

def BMMD_Naive(x,y,kernel):
    """
    Naive implementation of BMMD
    """
    lst = []
    for i in tqdm(range(x.shape[1])):
        lst.append(mmd_blockwise(x[:,i].to(device), y[:,i].to(device), kernel))
        # Call torch.cuda.empty_cache() to release GPU memory
        torch.cuda.empty_cache()
    return torch.tensor(lst).float()

def VDS_Naive(x,y,kernel):
    lst = []
    for i in tqdm(range(x.shape[-1])):
        idx = np.random.randint(0, x.shape[0]*x.shape[1], 10000)
        lst.append(MMD(x[:,:,i].flatten()[idx].unsqueeze(-1).cuda(),y[:,:,i].flatten()[idx].unsqueeze(-1).cuda(), kernel))
        torch.cuda.empty_cache()
    return torch.tensor(lst).float()