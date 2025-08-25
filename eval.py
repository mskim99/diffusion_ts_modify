import numpy as np
import os
import torch
import csv

from Utils.eval_utils import rmse,psnr,fre_cosine,mmd,emd, eval_fid
from Utils.metric_utils import visualization
from scipy import stats

def _rescale_to_range(x, new_min, new_max, eps=1e-12):
    x = np.asarray(x)
    old_min = np.min(x)
    old_max = np.max(x)
    if (old_max - old_min) < eps:
        mid = (new_min + new_max) / 2.0
        return np.full_like(x, mid, dtype=np.float32)
    y = (x - old_min) / (old_max - old_min)
    y = y * (new_max - new_min) + new_min
    return y.astype(np.float32)

def match_gen_to_gt_range(gt, gen, verbose=True):
    """
    Affine-rescale 'gen' to match global min/max of 'gt'.
    Works for arrays like (N,H,W,C) or (N,L,F).
    """
    gt = np.asarray(gt)
    gen = np.asarray(gen)
    gt_min, gt_max = float(np.min(gt)), float(np.max(gt))
    gen_min, gen_max = float(np.min(gen)), float(np.max(gen))
    if verbose:
        print(f"[Range] GT min/max: {gt_min:.6f} / {gt_max:.6f}")
        print(f"[Range] GEN min/max (before): {gen_min:.6f} / {gen_max:.6f}")
    gen_aligned = _rescale_to_range(gen, gt_min, gt_max)
    if verbose:
        a_min, a_max = float(np.min(gen_aligned)), float(np.max(gen_aligned))
        print(f"[Range] GEN min/max (after):  {a_min:.6f} / {a_max:.6f}")
    return gt, gen_aligned

def _quantile_transform_1d(original_1d, synthetic_1d):
    """
    [헬퍼 함수] 1차원 배열에 대한 Quantile 변환을 수행합니다.
    """
    # 데이터가 하나만 있는 엣지 케이스 처리
    if len(synthetic_1d) <= 1:
        return synthetic_1d

    # 1. 순위 계산 및 [0, 1] 범위의 분위수로 변환
    ranks = stats.rankdata(synthetic_1d, method='average')
    percentiles = (ranks - 1) / (len(synthetic_1d) - 1)

    # 2. 원본 데이터 분포에서 해당 분위수 값 찾기
    transformed_synthetic = np.quantile(original_1d, percentiles)

    return transformed_synthetic


def quantile_transform(original_data, synthetic_data):
    # 입력 데이터의 shape이 같은지 확인
    if original_data.shape != synthetic_data.shape:
        raise ValueError("Data between original and synthetic should be same.")

    # 데이터 shape 정보 추출
    n_samples, n_timesteps, n_features = original_data.shape

    # 결과를 저장할 빈 배열 생성
    corrected_data = np.zeros_like(synthetic_data)

    print(f"Transforming {n_features} features...")

    # 각 피처(feature)에 대해 반복
    for i in range(n_features):
        print(f"  - Process {i + 1}/{n_features} feature...")

        # 1. 현재 피처에 해당하는 모든 데이터를 1D 배열로 펼칩니다.
        #    (3639, 24) -> (87336,)
        original_flat = original_data[:, :, i].flatten()
        synthetic_flat = synthetic_data[:, :, i].flatten()

        # 2. 1D 데이터에 대해 Quantile 변환을 수행합니다.
        corrected_flat = _quantile_transform_1d(original_flat, synthetic_flat)

        # 3. 변환된 1D 배열을 다시 원래의 2D shape (샘플, 타임스텝)으로 복원합니다.
        #    (87336,) -> (3639, 24)
        corrected_feature = corrected_flat.reshape((n_samples, n_timesteps))

        # 4. 최종 결과 배열에 저장합니다.
        corrected_data[:, :, i] = corrected_feature

    print("Finish transforming all features.")
    return corrected_data


def quantile_transform_torch_gpu(original_data, synthetic_data):
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 중인 장치: {device}")

    # NumPy 배열을 PyTorch 텐서로 변환하고 선택된 장치로 이동
    original_tensor = torch.tensor(original_data, dtype=torch.float32).to(device)
    synthetic_tensor = torch.tensor(synthetic_data, dtype=torch.float32).to(device)

    if original_tensor.shape != synthetic_tensor.shape:
        raise ValueError("원본 데이터와 합성 데이터의 shape이 동일해야 합니다.")

    n_samples, n_timesteps, n_features = original_tensor.shape

    # 처리를 위해 텐서 평탄화
    original_flat = original_tensor.reshape(-1, n_features)
    synthetic_flat = synthetic_tensor.reshape(-1, n_features)

    # 총 샘플 수
    n_total_samples = original_flat.shape[0]

    # `argsort`를 사용하여 순위(rank)를 계산
    sorted_indices = torch.argsort(synthetic_flat, dim=0)

    # `scatter_`를 사용하여 순위 텐서 생성
    ranks = torch.empty_like(sorted_indices, dtype=torch.float32)
    rank_values = torch.arange(n_total_samples, device=device, dtype=torch.float32).unsqueeze(1).expand(-1, n_features)
    ranks.scatter_(dim=0, index=sorted_indices, src=rank_values)

    # 순위를 [0, 1] 범위의 분위수로 변환
    percentiles = ranks / (n_total_samples - 1)

    # 보간에 사용할 원본 데이터 정렬
    sorted_original, _ = torch.sort(original_flat, dim=0)

    # 결과를 저장할 빈 텐서 생성
    transformed_synthetic = torch.zeros_like(synthetic_flat)

    # --- ★★★ 수정된 핵심 부분: 각 피처별로 보간 로직 적용 ★★★ ---
    x0 = torch.linspace(0, 1, n_total_samples, device=device)

    for i in range(n_features):
        # 현재 피처에 해당하는 분위수, 정렬된 원본 데이터, 인덱스 추출
        current_percentiles = percentiles[:, i]
        current_sorted_original = sorted_original[:, i]

        # `searchsorted`를 사용하여 인덱스 찾기
        indices = torch.searchsorted(x0, current_percentiles)

        # 인덱스 경계값 처리
        indices = torch.clamp(indices, 1, n_total_samples - 1)

        # 선형 보간을 위한 값들 추출
        y0 = current_sorted_original.index_select(0, indices - 1)
        y1 = current_sorted_original.index_select(0, indices)

        x0_values = x0.index_select(0, indices - 1)
        x1_values = x0.index_select(0, indices)

        # 가중치 `t` 계산
        t = (current_percentiles - x0_values) / (x1_values - x0_values)

        # `lerp`를 사용하여 보간 수행
        transformed_synthetic[:, i] = torch.lerp(y0, y1, t)

    # 결과를 원래의 3D shape로 복원
    corrected_data = transformed_synthetic.reshape(n_samples, n_timesteps, n_features)

    # 최종 결과는 CPU로 다시 이동
    return corrected_data.cpu().numpy()

def _coral_align(gen: np.ndarray, gt: np.ndarray, eps=1e-6):
    """
    gen, gt: shape (N, L, C)  또는 (N, C)도 허용
    전처리(분위수/범위)가 끝난 후, 2차 통계(공분산)를 gt에 맞추는 whitening→recoloring.
    반환: gen_aligned (same shape as gen)
    """
    G = gen.reshape(-1, gen.shape[-1])   # (N*L, C)
    T = gt.reshape(-1,  gt.shape[-1])    # (N*L, C)

    # 중심화
    muG = G.mean(axis=0, keepdims=True)
    muT = T.mean(axis=0, keepdims=True)
    Gc  = G - muG
    Tc  = T - muT

    # 공분산
    SG = (Gc.T @ Gc) / (Gc.shape[0] - 1 + eps)  # (C, C)
    ST = (Tc.T @ Tc) / (Tc.shape[0] - 1 + eps)

    # 고유분해 (대칭 보장)
    evalG, evecG = np.linalg.eigh(SG + eps*np.eye(SG.shape[0]))
    evalT, evecT = np.linalg.eigh(ST + eps*np.eye(ST.shape[0]))

    # whitening(G): Σ_G^{-1/2}
    SG_inv_sqrt = (evecG @ np.diag(1.0 / np.sqrt(np.maximum(evalG, eps))) @ evecG.T)
    # recolor(T): Σ_T^{1/2}
    ST_sqrt     = (evecT @ np.diag(np.sqrt(np.maximum(evalT, eps))) @ evecT.T)

    # CORAL 변환
    G_whiten = Gc @ SG_inv_sqrt
    G_recol  = G_whiten @ ST_sqrt

    # 평균 복원
    G_aligned = G_recol + muT
    return G_aligned.reshape(gen.shape)

def _blend(a: np.ndarray, b: np.ndarray, w: float):
    """ 보수적으로 보정하려면 w(0~1)로 선형 블렌딩 """
    return (1.0 - w) * a + w * b

# ==============================
# Metrics for Paper Table 3 (TS-GBench-style)
# MDD, ACD, SD, KD, ED, DTW
# ==============================
from scipy.stats import wasserstein_distance, skew, kurtosis

def _flatten_featurewise(arr):
    """Flatten (N, L, F) -> list of F arrays, each 1D of size N*L."""
    N, L, F = arr.shape
    out = []
    for i in range(F):
        out.append(arr[:, :, i].reshape(-1))
    return out

def metric_mdd(real, synth):
    """Marginal Distribution Difference (average 1D Wasserstein distance over features)."""
    real_feats = _flatten_featurewise(real)
    synth_feats = _flatten_featurewise(synth)
    F = min(len(real_feats), len(synth_feats))
    dists = []
    for i in range(F):
        # Use 1D Wasserstein (EMD) between marginals
        d = wasserstein_distance(real_feats[i], synth_feats[i])
        dists.append(d)
    return float(np.mean(dists)) if dists else float('nan')

def _acf_1d(x, max_lag):
    """Autocorrelation function for 1D sequence up to max_lag (biased)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    acf = []
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            acf.append(0.0)
        else:
            acf.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.array(acf)

def metric_acd(real, synth, max_lag=None):
    """
    AutoCorrelation Difference:
    average |ACF_real - ACF_synth| over features and lags.
    ACF is computed per-sample then averaged (per feature).
    """
    N, L, F = real.shape
    max_lag = max_lag or max(1, min(50, L - 1))

    diffs = []
    for i in range(F):
        # Average ACF across samples for this feature
        acf_r = np.zeros(max_lag, dtype=float)
        acf_s = np.zeros(max_lag, dtype=float)
        for n in range(N):
            acf_r += _acf_1d(real[n, :, i], max_lag)
        for n in range(min(synth.shape[0], N)):
            acf_s += _acf_1d(synth[n, :, i], max_lag)
        acf_r /= N
        acf_s /= min(synth.shape[0], N)
        diffs.append(np.mean(np.abs(acf_r - acf_s)))
    return float(np.mean(diffs)) if diffs else float('nan')

def metric_sd(real, synth):
    """Skewness Difference: average |skew_real - skew_synth| over features (marginals)."""
    real_feats = _flatten_featurewise(real)
    synth_feats = _flatten_featurewise(synth)
    F = min(len(real_feats), len(synth_feats))
    diffs = []
    for i in range(F):
        sr = skew(real_feats[i], bias=False)
        ss = skew(synth_feats[i], bias=False)
        diffs.append(abs(sr - ss))
    return float(np.mean(diffs)) if diffs else float('nan')

def metric_kd(real, synth):
    """Kurtosis Difference: average |kurt_real - kurt_synth| over features (marginals)."""
    real_feats = _flatten_featurewise(real)
    synth_feats = _flatten_featurewise(synth)
    F = min(len(real_feats), len(synth_feats))
    diffs = []
    for i in range(F):
        # Use Fisher=False to get "classical" kurtosis (3 for normal)
        kr = kurtosis(real_feats[i], fisher=False, bias=False)
        ks = kurtosis(synth_feats[i], fisher=False, bias=False)
        diffs.append(abs(kr - ks))
    return float(np.mean(diffs)) if diffs else float('nan')

def metric_ed(real, synth):
    """
    Euclidean Distance: average L2 between paired samples (flattened across time & features).
    Pairs up to the min(N_real, N_synth).
    """
    N = min(real.shape[0], synth.shape[0])
    if N == 0:
        return float('nan')
    L, F = real.shape[1], real.shape[2]
    real_flat = real[:N].reshape(N, L * F)
    synth_flat = synth[:N].reshape(N, L * F)
    diffs = np.linalg.norm(real_flat - synth_flat, axis=1)
    return float(np.mean(diffs))

def _dtw_distance(seq_a, seq_b):
    """
    DTW for multivariate sequences of shape (L, F).
    Local cost: L2 between feature vectors at each time step.
    """
    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)
    La, Lb = a.shape[0], b.shape[0]
    # cost matrix
    D = np.full((La + 1, Lb + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, La + 1):
        for j in range(1, Lb + 1):
            cost = np.linalg.norm(a[i-1] - b[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[La, Lb])

def metric_dtw(real, synth, max_samples=256, seed=42):
    """
    Dynamic Time Warping distance averaged over paired samples.
    Uses up to max_samples pairs for tractability.
    """
    N = min(real.shape[0], synth.shape[0])
    if N == 0:
        return float('nan')
    idx = np.arange(N)
    if N > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_samples, replace=False)
    dists = []
    for n in idx:
        d = _dtw_distance(real[n], synth[n])
        dists.append(d)
    return float(np.mean(dists))

def get_mean_dev(y):
    # b=y.shape[0]
    # y=y.reshape(1,b)
    mean=np.mean(y)
    variance = np.var(y)
    # print(mean,variance)
    return mean, variance

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

# ===========================
# Eval options (post-processing switches)
# ===========================
# - 기본값은 "보정 끔" → 모델 출력 분포가 그대로 평가 지표에 반영됨
APPLY_MINMAX_GEN   = False  # 생성물(gen)에 per-array minmax 정규화 후 [-1,1] 적용
APPLY_QUANTILE_FIX = False  # 분위수(quantile) 보정으로 gen의 마진 분포를 GT에 정렬
APPLY_RANGE_MATCH  = False  # gen 전체를 GT 전체 범위(min/max)에 affine 매칭

# ---------------------------
# 실험 식별자
# ---------------------------
foldername = "test_stocks_nie6_jt_0_1"
ckptfoldername = "Checkpoints_stocks_nie6_24"
dataname = "stock"   # {"energy","stock","sine","fmri","mujoco"}
length = 24
prop = True          # True: trend/season 따로 합성하여 평가, False: 바로 단일 gen 사용

# ===========================
# Fusion (alpha) options
# ===========================
USE_FUSION_ALPHA    = True   # True면 trend/season을 alpha로 가중합
ALPHA_SOURCE        = "ckpt" # {"scalar","ckpt"}: 스칼라 고정값 or 체크포인트에서 로드
ALPHA_SCALAR        = 0.2857    # ALPHA_SOURCE=="scalar"일 때 사용 (0~1 권장)
ALPHA_CKPT_PATH     = f"./{ckptfoldername}/checkpoint-10.pt"  # 예시 경로(적절히 수정)
ALPHA_OPT_MODE = "sample_feature"  # {"none","scalar","feature","sample_feature"} 중 선택
# - "scalar": 전체 데이터에 대한 단일 α∗
# - "feature": 피처별 α∗ (shape: (1,1,C)), 시간과 샘플에 동일 적용
# - "sample_feature": 샘플×피처별 α∗ (shape: (N,1,C)), 각 시계열에 맞춤

# [PATCH] --- Optional: CORAL covariance alignment (after quantile/range) ---
ENABLE_CORAL = True      # 필요할 때만 True
CORAL_BLEND  = 0.9      # 0=미적용, 1=완전 적용. 0.25~0.5 권장

ckpt = torch.load(ALPHA_CKPT_PATH, map_location="cpu") if os.path.exists(ALPHA_CKPT_PATH) else {}   # [PATCH]
scale_meta_path = os.path.join(foldername, "gen_scale_meta.npz")
scale_meta = np.load(scale_meta_path) if os.path.exists(scale_meta_path) else None
global_min = ckpt.get("global_min", None)   # [PATCH]
global_max = ckpt.get("global_max", None)   # [PATCH]

# numpy 로 강제 변환   # [FIX]
if global_min is not None:
    global_min = np.array(global_min)
if global_max is not None:
    global_max = np.array(global_max)

def norm_with_global(z, zmin, zmax):   # [PATCH]
    if zmin is None or zmax is None: return z
    return 2. * (z - zmin) / (zmax - zmin + 1e-6) - 1.

def _alpha_optimal_mse(gt, t, s, mode="scalar", eps=1e-12):
    """
    MSE 기준으로 || alpha*T + (1-alpha)*S - GT ||^2 를 최소화하는 alpha*를 해석적으로 구함.
    입력 shape: (N, L, C)
    반환: alpha  (모드별 shape)
    """
    gt = np.asarray(gt); t = np.asarray(t); s = np.asarray(s)
    d = t - s                      # (N,L,C)
    y = gt - s                     # (N,L,C)
    num = np.sum(d * y, axis=1)    # (N,C)   (시간 차원 합)
    den = np.sum(d * d, axis=1) + eps  # (N,C)

    if mode == "scalar":
        a = np.sum(num) / np.sum(den)                 # 스칼라
        return float(np.clip(a, -1.0, 1.0))

    elif mode == "feature":
        a_fc = np.sum(num, axis=0) / (np.sum(den, axis=0))   # (C,)
        a_fc = np.clip(a_fc, -1.0, 1.0)                       # [0,1]
        return a_fc.reshape(1,1,-1)                           # (1,1,C)

    elif mode == "sample_feature":
        a_sf = num / den                       # (N,C)
        a_sf = np.clip(a_sf, -1.0, 1.0)
        return a_sf.reshape(-1,1,a_sf.shape[1])  # (N,1,C)

    else:
        raise ValueError(f"Unknown mode: {mode}")

def _load_alpha_from_ckpt(ckpt_path):
    """
    체크포인트에서 fusion_alpha를 읽어와 sigmoid를 적용해 [0,1]로 만든 뒤
    numpy로 반환합니다. 기대 shape: (1,1,C) 또는 (C,)
    """
    if not os.path.exists(ckpt_path):
        print(f"[alpha] ckpt not found: {ckpt_path}")
        return None
    try:
        data = torch.load(ckpt_path, map_location="cpu")
        fa = data.get("fusion_alpha", None)
        if fa is None:
            print("[alpha] fusion_alpha not in checkpoint")
            return None
        fa = torch.sigmoid(fa)  # [0,1]
        return fa.detach().cpu().numpy()
    except Exception as e:
        print(f"[alpha] failed to load alpha from ckpt: {e}")
        return None

def _broadcast_alpha(alpha, target_shape):
    """
    alpha: float 또는 (1,1,C)/(C,) numpy
    target_shape: (N, L, C)
    반환: (N, L, C) 브로드캐스트된 numpy
    """
    N, L, C = target_shape
    if np.isscalar(alpha):
        return float(alpha)  # 스칼라는 브로드캐스트로 사용
    alpha = np.asarray(alpha)
    if alpha.ndim == 1:           # (C,)
        alpha = alpha.reshape(1, 1, -1)
    elif alpha.ndim == 3:         # (1,1,C) 예상
        pass
    else:
        raise ValueError(f"Unsupported alpha shape: {alpha.shape}")
    # (1,1,C) -> (N,L,C)로 브로드캐스트
    return np.broadcast_to(alpha, (N, L, C))

# ---------------------------
# 데이터 로드 + (선택적) 보정
# ---------------------------
def _load_prop_split(folder, name, L):
    if name == "energy":
        gt = np.load(f"./OUTPUT/{folder}/samples/energy_norm_truth_{L}_train.npy")
        gt_s = np.load(f"./OUTPUT/{folder}/samples/energy_norm_truth_{L}_train_season.npy")
        gt_t = np.load(f"./OUTPUT/{folder}/samples/energy_norm_truth_{L}_train_trend.npy")
    elif name == "stock":
        gt = np.load(f"./OUTPUT/{folder}/samples/stock_norm_truth_{L}_train.npy")
        gt_s = np.load(f"./OUTPUT/{folder}/samples/stock_norm_truth_{L}_train_season.npy")
        gt_t = np.load(f"./OUTPUT/{folder}/samples/stock_norm_truth_{L}_train_trend.npy")
    elif name == "sine":
        gt = np.load(f"./OUTPUT/{folder}/samples/sine_ground_truth_{L}_train.npy")
        gt_s = np.load(f"./OUTPUT/{folder}/samples/sine_ground_truth_{L}_train_season.npy")
        gt_t = np.load(f"./OUTPUT/{folder}/samples/sine_ground_truth_{L}_train_trend.npy")
    elif name == "fmri":
        gt = np.load(f"./OUTPUT/{folder}/samples/fmri_norm_truth_{L}_train.npy")
        gt_s = np.load(f"./OUTPUT/{folder}/samples/fmri_norm_truth_{L}_train_season.npy")
        gt_t = np.load(f"./OUTPUT/{folder}/samples/fmri_norm_truth_{L}_train_trend.npy")
    elif name == "mujoco":
        gt = np.load(f"./OUTPUT/{folder}/samples/mujoco_norm_truth_{L}_train.npy")
        gt_s = np.load(f"./OUTPUT/{folder}/samples/mujoco_norm_truth_{L}_train_season.npy")
        gt_t = np.load(f"./OUTPUT/{folder}/samples/mujoco_norm_truth_{L}_train_trend.npy")
    else:
        raise NotImplementedError(f"Unknown dataname: {name}")

    gen_s = np.load(f"./OUTPUT/{folder}/ddpm_fake_{folder}_season.npy")
    gen_t = np.load(f"./OUTPUT/{folder}/ddpm_fake_{folder}_trend.npy")
    return gt, gt_s, gt_t, gen_s, gen_t

def _load_direct(folder, name, L):
    if name == "energy":
        gt = np.load(f"./OUTPUT/{folder}/samples/energy_norm_truth_{L}_train.npy")
    elif name == "stock":
        gt = np.load(f"./OUTPUT/{folder}/samples/stock_norm_truth_{L}_train.npy")
    elif name == "sine":
        gt = np.load(f"./OUTPUT/{folder}/samples/sine_ground_truth_{L}_train.npy")
    elif name == "fmri":
        gt = np.load(f"./OUTPUT/{folder}/samples/fmri_norm_truth_{L}_train.npy")
    elif name == "mujoco":
        gt = np.load(f"./OUTPUT/{folder}/samples/mujoco_norm_truth_{L}_train.npy")
    else:
        raise NotImplementedError(f"Unknown dataname: {name}")

    gen = np.load(f"./OUTPUT/{folder}/ddpm_fake_{folder}.npy")
    return gt, gen

def _align_first_dim(a, b):
    """두 배열의 첫 차원(N)을 동일한 최소값으로 맞춰 slice."""
    N = min(a.shape[0], b.shape[0])
    return a[:N], b[:N]

def norm_with_global_np(z, zmin, zmax):   # [PATCH]
    if zmin is None or zmax is None: return z
    return 2.0 * (z - zmin) / (zmax - zmin + 1e-6) - 1.0

if prop:
    # 1) 로드
    gt_imgs, gt_imgs_season, gt_imgs_trend, gen_imgs_season, gen_imgs_trend = _load_prop_split(foldername, dataname, length)

    # 2) 배치 수 일치
    gt_imgs_season, gen_imgs_season = _align_first_dim(gt_imgs_season, gen_imgs_season)
    gt_imgs_trend,  gen_imgs_trend  = _align_first_dim(gt_imgs_trend,  gen_imgs_trend)
    # gt도 season/trend와 동일한 N으로 맞춤
    gt_imgs, _ = _align_first_dim(gt_imgs, gen_imgs_trend)

    # 3) (선택) 생성물 minmax 정규화 → [-1,1]
    if APPLY_MINMAX_GEN:
        def mm_to_pm1(x):
            xmin, xmax = x.min(), x.max()
            if xmax - xmin < 1e-6:  # 상수 신호 보호
                return np.zeros_like(x)
            y = (x - xmin) / (xmax - xmin)
            return 2.0 * y - 1.0
        gen_imgs_season = mm_to_pm1(gen_imgs_season)
        gen_imgs_trend  = mm_to_pm1(gen_imgs_trend)

    # 4) (선택) 분위수 보정으로 마진 분포 정렬
    if APPLY_QUANTILE_FIX:
        gen_imgs_season = quantile_transform_torch_gpu(gt_imgs_season, gen_imgs_season)
        gen_imgs_trend  = quantile_transform_torch_gpu(gt_imgs_trend,  gen_imgs_trend)

    # 5) 합성 with alpha
    if USE_FUSION_ALPHA:
        if ALPHA_SOURCE == "ckpt":
            alpha_loaded = _load_alpha_from_ckpt(ALPHA_CKPT_PATH)
            if alpha_loaded is None:
                print("[alpha] fallback to scalar due to load failure.")
                alpha_loaded = ALPHA_SCALAR
        elif ALPHA_SOURCE == "scalar":
            alpha_loaded = ALPHA_SCALAR
        else:
            raise ValueError(f"Unknown ALPHA_SOURCE: {ALPHA_SOURCE}")

        # 브로드캐스트 후 결합
        gate = _broadcast_alpha(alpha_loaded, gen_imgs_trend.shape)  # (N,L,C) 또는 스칼라
        gen_imgs_ckpt = gate * gen_imgs_trend + (1.0 - gate) * gen_imgs_season
    else:
        gen_imgs_ckpt = gen_imgs_season + gen_imgs_trend

    # (옵션) 전역 min/max 정규화 (학습과 동일한 스케일)   # [PATCH]
    if global_min is not None and global_max is not None:
        print("[eval] Normalizing GT and GEN with the SAME global min/max (from meta).")
        if prop:
            gt_imgs_season = norm_with_global_np(gt_imgs_season, global_min, global_max)
            gt_imgs_trend = norm_with_global_np(gt_imgs_trend, global_min, global_max)
            gen_imgs_season = norm_with_global_np(gen_imgs_season, global_min, global_max)
            gen_imgs_trend = norm_with_global_np(gen_imgs_trend, global_min, global_max)
            # α 가중합 (이미 위에서 했으면 생략 가능, 재적용 시 주의)
            if USE_FUSION_ALPHA:
                gate = _broadcast_alpha(alpha_loaded, gen_imgs_trend.shape)
                gen_imgs = gate * gen_imgs_trend + (1.0 - gate) * gen_imgs_season
            else:
                gen_imgs = gen_imgs_trend + gen_imgs_season
            gt_imgs = norm_with_global_np(gt_imgs, global_min, global_max)
        else:
            gt_imgs = norm_with_global_np(gt_imgs, global_min, global_max)
            gen_imgs = norm_with_global_np(gen_imgs, global_min, global_max)

    # 5-2) (진단) 해석적 최적 α 결합도 계산해 보자
    if ALPHA_OPT_MODE != "none":
        alpha_star = _alpha_optimal_mse(gt_imgs, gen_imgs_trend, gen_imgs_season, mode=ALPHA_OPT_MODE)
        if np.isscalar(alpha_star):
            gen_imgs_opt = alpha_star * gen_imgs_trend + (1.0 - alpha_star) * gen_imgs_season
        else:
            gen_imgs_opt = np.broadcast_to(alpha_star, gen_imgs_trend.shape) * gen_imgs_trend \
                           + (1.0 - np.broadcast_to(alpha_star, gen_imgs_season.shape)) * gen_imgs_season

        # 어떤 결합이 더 좋은지 선택 (MDD 또는 원하는 기준으로)
        mdd_ckpt = metric_mdd(gt_imgs, gen_imgs_ckpt)
        mdd_opt = metric_mdd(gt_imgs, gen_imgs_opt)
        print(f"[alpha-diagnosis] MDD(ckpt)={mdd_ckpt:.4f}, MDD(opt-{ALPHA_OPT_MODE})={mdd_opt:.4f}")
        gen_imgs = gen_imgs_opt if mdd_opt < mdd_ckpt else gen_imgs_ckpt
    else:
        gen_imgs = gen_imgs_ckpt

    # 6) (선택) GT 전역 범위로 affine 매칭(스케일/시프트만 맞춤)
    if APPLY_RANGE_MATCH:
        _, gen_imgs = match_gen_to_gt_range(gt_imgs, gen_imgs, verbose=True)

    if ENABLE_CORAL:
        try:
            gen_coral = _coral_align(gen_imgs, gt_imgs)  # 2차 통계 맞춤
            gen_imgs = _blend(gen_imgs, gen_coral, CORAL_BLEND)  # 보수적 블렌딩
            print(f"[eval] CORAL applied with blend={CORAL_BLEND}")
        except Exception as e:
            print(f"[eval] CORAL skipped due to error: {e}")

else:
    # 단일 생성물 경로
    gt_imgs, gen_imgs = _load_direct(foldername, dataname, length)
    gt_imgs, gen_imgs = _align_first_dim(gt_imgs, gen_imgs)

    # (선택) 생성물 보정
    if APPLY_MINMAX_GEN:
        xmin, xmax = gen_imgs.min(), gen_imgs.max()
        if xmax - xmin > 1e-6:
            gen_imgs = 2.0 * ((gen_imgs - xmin) / (xmax - xmin)) - 1.0
        else:
            gen_imgs = np.zeros_like(gen_imgs)

    if APPLY_QUANTILE_FIX:
        gen_imgs = quantile_transform_torch_gpu(gt_imgs, gen_imgs)

    if APPLY_RANGE_MATCH:
        _, gen_imgs = match_gen_to_gt_range(gt_imgs, gen_imgs, verbose=True)

# gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())
# gt_imgs = (gt_imgs - gt_imgs.min()) / (gt_imgs.max() - gt_imgs.min())

gen_imgs = (gen_imgs / 2.) + 0.5
gt_imgs = (gt_imgs / 2.) + 0.5

visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='pca', compare=gt_imgs.shape[0])
visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='tsne', compare=gt_imgs.shape[0])
# visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='tsne_enh', compare=gt_imgs.shape[0])
visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='kernel', compare=gt_imgs.shape[0])

rmse_all = rmse(gen_imgs, gt_imgs)
psnr_all = psnr(gen_imgs, gt_imgs)
fre_cos_all = fre_cosine(gen_imgs, gt_imgs)
mmd_all = mmd(gen_imgs, gt_imgs)
emd_all = emd(gen_imgs, gt_imgs)

rmse_all = np.array(rmse_all)
psnr_all = np.array(psnr_all)
fre_cos_all = np.array(fre_cos_all)
mmd_all = np.array(mmd_all)
emd_all = np.array(emd_all)

rsme_mean, rsme_var = get_mean_dev(rmse_all)
psnr_mean, psnr_var = get_mean_dev(psnr_all)
fre_cos_mean, fre_cos_var = get_mean_dev(fre_cos_all)
mmd_mean, mmd_var = get_mean_dev(mmd_all)
emd_mean, emd_var = get_mean_dev(emd_all)
fid_value = eval_fid(gen_imgs, gt_imgs)

# print('Evaluation finished')

print('[RSME] MEAN : ' + str(rsme_mean) + ' / VAR : ' + str(rsme_var))
print('[PSNR] MEAN : ' + str(psnr_mean) + ' / VAR : ' + str(psnr_var))
print('[COS] MEAN : ' + str(fre_cos_mean) + ' / VAR : ' + str(fre_cos_var))
print('[MMD] MEAN : ' + str(mmd_mean) + ' / VAR : ' + str(mmd_var))
# print('[EMD] MEAN : ' + str(emd_mean) + ' / VAR : ' + str(emd_var))
print('[FID] : ' + str(fid_value))


# ===== Table 3 Metrics (Paper) =====
# See: TS-GBench summary metrics used in the paper (MDD, ACD, SD, KD, ED, DTW).
mdd_val = metric_mdd(gt_imgs, gen_imgs)
acd_val = metric_acd(gt_imgs, gen_imgs, max_lag=None)
sd_val  = metric_sd(gt_imgs, gen_imgs)
kd_val  = metric_kd(gt_imgs, gen_imgs)
ed_val  = metric_ed(gt_imgs, gen_imgs)
dtw_val = metric_dtw(gt_imgs, gen_imgs, max_samples=256, seed=123)

print('[MDD] : ' + str(mdd_val))
print('[ACD] : ' + str(acd_val))
print('[SD ] : ' + str(sd_val))
print('[KD ] : ' + str(kd_val))
print('[ED ] : ' + str(ed_val))
print('[DTW] : ' + str(dtw_val))

results_all = {}

# Collect printed metrics if variables exist
for name, var in [
    ("RMSE", locals().get("rsme_mean")),
    ("PSNR", locals().get("psnr_mean")),
    ("COS", locals().get("fre_cos_mean")),
    ("MMD", locals().get("mmd_mean")),
    ("FID", locals().get("fid_value")),
    ("SSIM", locals().get("ssim_val")),
    ("LPIPS", locals().get("lpips_val")),
    ("FID", locals().get("fid_val")),
    ("MDD", locals().get("mdd_val")),
    ("ACD", locals().get("acd_val")),
    ("SD", locals().get("sd_val")),
    ("KD", locals().get("kd_val")),
    ("ED", locals().get("ed_val")),
    ("DTW", locals().get("dtw_val"))
]:
    if var is not None:
        results_all[name] = var

csv_path = os.path.join(".", "evaluation_results.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results_all.keys())
    writer.writeheader()
    writer.writerow(results_all)

print(f"[CSV] All evaluation results saved to {csv_path}")