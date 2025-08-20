import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

warnings.filterwarnings("ignore")
import numpy as np

from eval_utils.metric_utils import display_scores
from eval_utils.discriminative_metric import discriminative_score_metrics
from eval_utils.predictive_metric import predictive_score_metrics
import torch
from eval_utils.MMD import BMMD, cross_correlation_distribution, BMMD_Naive, VDS_Naive
from scipy import stats


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


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5

def discriminative_score(ori_data, fake_data, iterations=5):
    fake_data = fake_data[: ori_data.shape[0]]
    discriminative_score = []
    print(f"Fake data:", "min ", fake_data.min(), ", max ", fake_data.max())
    print(f"Real data:", "min ", ori_data.min(), ", max ", ori_data.max())
    for i in range(iterations):
        print(i)
        temp_disc, fake_acc, real_acc, values = discriminative_score_metrics(
            ori_data[:], fake_data[: ori_data.shape[0]]
        )
        discriminative_score.append(temp_disc)
        print(f"Iter {i}: ", temp_disc, ",", fake_acc, ",", real_acc, "\n")
    print(f"{dataname}:")
    display_scores(discriminative_score)

    return values


def predictive_score(ori_data, fake_data, iterations=5):
    fake_data = fake_data[: ori_data.shape[0]]
    predictive_score = []
    print(f"Fake data:", "min ", fake_data.min(), ", max ", fake_data.max())
    print(f"Real data:", "min ", ori_data.min(), ", max ", ori_data.max())
    for i in range(iterations):
        temp_pred = predictive_score_metrics(ori_data, fake_data[: ori_data.shape[0]])
        predictive_score.append(temp_pred)
        print(i, " epoch: ", temp_pred, "\n")
    print(f"{dataname}:")
    display_scores(predictive_score)


def BMMD_score(ori_data, fake_data):
    fake_data = fake_data[: ori_data.shape[0]]
    ori_data = torch.tensor(ori_data).float()
    fake_data = torch.tensor(fake_data).float()

    ori_data = cross_correlation_distribution(ori_data).unsqueeze(-1).permute(1, 0, 2)
    fake_data = cross_correlation_distribution(fake_data).unsqueeze(-1).permute(1, 0, 2)

    assert ori_data.shape == fake_data.shape

    mmd_loss = BMMD(ori_data, fake_data, "rbf").mean()
    print(f"{dataname}:", mmd_loss)


def BMMD_score_naive(ori_data, fake_data):
    fake_data = fake_data[: ori_data.shape[0]]
    ori_data = torch.tensor(ori_data).float()
    fake_data = torch.tensor(fake_data).float()

    ori_data = cross_correlation_distribution(ori_data).unsqueeze(-1)
    fake_data = cross_correlation_distribution(fake_data).unsqueeze(-1)

    assert ori_data.shape == fake_data.shape

    mmd_loss = BMMD_Naive(ori_data, fake_data, "rbf").mean()
    print(f"{dataname} FDDS Score:", mmd_loss)


def VDS_score(ori_data, fake_data):
    fake_data = fake_data[: ori_data.shape[0]]
    ori_data = torch.tensor(ori_data).float()
    fake_data = torch.tensor(fake_data).float()

    vds_score = VDS_Naive(ori_data, fake_data, "rbf").mean()
    print(f"{dataname} VDS Score:", vds_score)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    foldername = "test_energy"
    dataname = "energy"
    length = 24
    prop = False

    if prop:
        if dataname == "energy":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/energy_norm_truth_{length}_train.npy")
            ori_season = np.load(f"./OUTPUT/{foldername}/samples/energy_norm_truth_{length}_train_season.npy")
            ori_trend = np.load(f"./OUTPUT/{foldername}/samples/energy_norm_truth_{length}_train_trend.npy")
            season_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_season.npy")
            trend_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_trend.npy")
        elif dataname == "stock":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/stock_norm_truth_{length}_train.npy")
            ori_season = np.load(f"./OUTPUT/{foldername}/samples/stock_norm_truth_{length}_train_season.npy")
            ori_trend = np.load(f"./OUTPUT/{foldername}/samples/stock_norm_truth_{length}_train_trend.npy")
            season_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_season.npy")
            trend_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_trend.npy")
        elif dataname == "sine":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/sine_ground_truth_{length}_train.npy")
            ori_season = np.load(f"./OUTPUT/{foldername}/samples/sine_ground_truth_{length}_train_season.npy")
            ori_trend = np.load(f"./OUTPUT/{foldername}/samples/sine_ground_truth_{length}_train_trend.npy")
            season_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_season.npy")
            trend_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_trend.npy")
        elif dataname == "fmri":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/fmri_norm_truth_{length}_train.npy")
            ori_season = np.load(f"./OUTPUT/{foldername}/samples/fmri_norm_truth_{length}_train_season.npy")
            ori_trend = np.load(f"./OUTPUT/{foldername}/samples/fmri_norm_truth_{length}_train_trend.npy")
            season_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_season.npy")
            trend_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_trend.npy")
        elif dataname == "mujoco":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/mujoco_norm_truth_{length}_train.npy")
            ori_season = np.load(f"./OUTPUT/{foldername}/samples/mujoco_norm_truth_{length}_train_season.npy")
            ori_trend = np.load(f"./OUTPUT/{foldername}/samples/mujoco_norm_truth_{length}_train_trend.npy")
            season_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_season.npy")
            trend_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}_trend.npy")
        else:
            raise NotImplementedError(f"Unkown dataname: {dataname}")

        season_data = (season_data - season_data.min()) / (season_data.max() - season_data.min())
        season_data = 2. * season_data - 1.

        trend_data = (trend_data - trend_data.min()) / (trend_data.max() - trend_data.min())
        trend_data = 2. * trend_data - 1.

        if season_data.shape[0] > ori_season.shape[0]:
            season_data = season_data[0:ori_season.shape[0], :, :]
            trend_data = trend_data[0:ori_trend.shape[0], :, :]
        else:
            ori_season = ori_season[0:season_data.shape[0], :, :27]
            ori_trend = ori_trend[0:trend_data.shape[0], :, :27]
            ori_data = ori_data[0:trend_data.shape[0], :, :27]

        season_data = quantile_transform_torch_gpu(ori_season, season_data)
        trend_data = quantile_transform_torch_gpu(ori_trend, trend_data)

        gen_data = season_data + trend_data

        gen_data = (gen_data / 2.) + 0.5
        # ori_data = (ori_data / 2.) + 0.5
    else:
        if dataname == "energy":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/energy_norm_truth_{length}_train.npy")
            gen_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}.npy")
        elif dataname == "stock":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/stock_norm_truth_{length}_train.npy")
            gen_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}.npy")
        elif dataname == "sine":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/sine_ground_truth_{length}_train.npy")
            gen_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}.npy")
        elif dataname == "fmri":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/fmri_norm_truth_{length}_train.npy")
            gen_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}.npy")
        elif dataname == "mujoco":
            ori_data = np.load(f"./OUTPUT/{foldername}/samples/mujoco_norm_truth_{length}_train.npy")
            gen_data = np.load(f"./OUTPUT/{foldername}/ddpm_fake_{foldername}.npy")

        if gen_data.shape[0] > ori_data.shape[0]:
            gen_data = gen_data[0:ori_data.shape[0], :, :]
        else:
            ori_data = ori_data[0:gen_data.shape[0], :, :]

    print(ori_data.min())
    print(ori_data.max())
    print(gen_data.min())
    print(gen_data.max())

    print('VDS score')
    VDS_score(ori_data, gen_data)
    print('discriminative score (DA)')
    discriminative_score(ori_data, gen_data)
    print('predictive score')
    predictive_score(ori_data, gen_data)
    print('BMMD score (naive / FDDS)')
    BMMD_score_naive(ori_data, gen_data)
