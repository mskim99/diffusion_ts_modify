import numpy as np
import os
import glob
import random

from Utils.eval_utils import rmse,psnr,fre_cosine,mmd,emd, eval_fid
from Utils.metric_utils import visualization

from scipy import stats
from sklearn.neighbors import KernelDensity


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

# Case 1 : diffwave
'''
gen_paths = glob.glob('/data/jionkim/Test/diffwave/*.npy')
gt_paths = glob.glob('/data/jionkim/Test/diffwave_gt/*.npy')

gen_imgs = []
gt_imgs = []

idx = 0
for name in gen_paths:
    img = np.load(name)
    gen_imgs.append(img)
    idx = idx+1

idx = 0
for name in gt_paths:
    img = np.load(name)

    # img = (img - img.min()) / (img.max() - img.min())
    # img = 2. * img - 1.
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std

    gt_imgs.append(img)
    idx = idx+1

gen_imgs = np.array(gen_imgs)
gt_imgs = np.array(gt_imgs)

gen_imgs = gen_imgs[:, 0, 0:41216]
gt_imgs = gt_imgs[:,  0:41216]
print(gen_imgs.shape)
print(gt_imgs.shape)
'''

# Case 2 : diffusion_TS
# gen_imgs = np.load('/data/jionkim/diffusion_TS_modify/OUTPUT/test_stocks_nie3_lr_1e_5/ddpm_fake_test_stocks_nie3_lr_1e_5.npy')
gen_imgs_season = np.load('/data/jionkim/diffusion_TS_modify/OUTPUT/test_stocks_nie_st_sap_lr_1e_5/ddpm_fake_test_stocks_nie_st_sap_lr_1e_5_season.npy')
gen_imgs_trend = np.load('/data/jionkim/diffusion_TS_modify/OUTPUT/test_stocks_nie_st_sap_lr_1e_5/ddpm_fake_test_stocks_nie_st_sap_lr_1e_5_trend.npy')
gt_imgs = np.load('/data/jionkim/diffusion_TS_modify/OUTPUT/test_stocks_nie_st_sap_lr_1e_5/samples/stock_norm_truth_24_train.npy')

print(gen_imgs_season.min())
print(gen_imgs_season.max())
print(gen_imgs_trend.min())
print(gen_imgs_trend.max())

gen_imgs_season = (gen_imgs_season - gen_imgs_season.min()) / (gen_imgs_season.max() - gen_imgs_season.min())
gen_imgs_season = 2. * gen_imgs_season - 1.
gen_imgs_season = gen_imgs_season[0:3639, :, :]

gen_imgs_trend = (gen_imgs_trend - gen_imgs_trend.min()) / (gen_imgs_trend.max() - gen_imgs_trend.min())
gen_imgs_trend = 2. * gen_imgs_trend - 1.
gen_imgs_trend = gen_imgs_trend[0:3639, :, :]

gen_imgs = gen_imgs_season + gen_imgs_trend

print(gen_imgs.shape)
print(gen_imgs.min())
print(gen_imgs.max())
print(gt_imgs.min())
print(gt_imgs.max())

gen_imgs = quantile_transform(gt_imgs, gen_imgs)

# mean = np.mean(gen_imgs, axis=1, keepdims=True)
# std = np.std(gen_imgs, axis=1, keepdims=True)  # 표준편차
# gen_imgs = (gen_imgs - mean) / std

# gen_imgs = 2. * gen_imgs - 1.
# gt_imgs = 2. * gt_imgs - 1.

# gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())
# gen_imgs = 2. * gen_imgs - 1.

# gen_imgs = gen_imgs[0:24, :, :]
# gen_imgs = np.load('/data/jionkim/diffusion_TS/OUTPUT/test_stock/ddpm_fake_test_stock.npy')
# gt_imgs = np.load('/data/jionkim/diffusion_TS/OUTPUT/test_stock/samples/stock_norm_truth_24_train.npy')
# gt_imgs_path = glob.glob('/data/jionkim/diffusion_TS/Data/datasets/textum_accum_48k_5/*.npy')
'''
random.shuffle(gt_imgs_path)

gt_imgs = []
for i in range (0, 16):
    print(gt_imgs_path[i])
    gt_img = np.load(gt_imgs_path[i])
    gt_imgs.append(gt_img)
gt_imgs = np.array(gt_imgs)
'''

# gen_imgs = gen_imgs.reshape([16, -1, 5])

'''
mean = np.mean(gen_imgs)
std = np.std(gen_imgs)
# gen_imgs = (gen_imgs - mean) / std

mean = np.mean(gt_imgs)
std = np.std(gt_imgs)
# gt_imgs = (gt_imgs - mean) / std
'''

# gen_imgs = gen_imgs[0:3430, 0:2, :]
# gt_imgs = gt_imgs[0:3430, 0:2, :]

# gen_imgs = np.reshape(gen_imgs, (48000, -1))
# gt_imgs = np.reshape(gt_imgs, (48000, -1))

# gen_imgs = np.swapaxes(gen_imgs, 1, 0)
# gt_imgs = np.swapaxes(gt_imgs, 1, 0)

# gen_imgs = (gen_imgs - gen_imgs.min()) / (gen_imgs.max() - gen_imgs.min())

# gen_imgs = np.expand_dims(gen_imgs, 1)
# gt_imgs = np.expand_dims(gt_imgs, 1)

# gen_imgs = np.concatenate([gen_imgs, gen_imgs], axis=1)
# gt_imgs = np.concatenate([gt_imgs, gt_imgs], axis=1)

# gen_imgs = np.swapaxes(gen_imgs, 1, 0)
# gt_imgs = np.swapaxes(gt_imgs, 1, 0)


print(gen_imgs.shape)
print(gt_imgs.shape)

visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='pca', compare=gt_imgs.shape[0])
visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='tsne', compare=gt_imgs.shape[0])
visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='kernel', compare=gt_imgs.shape[0])

# gen_imgs = torch.Tensor(gen_imgs).cuda()
# gt_imgs_shuffle = torch.Tensor(gt_imgs_shuffle).cuda()

gen_imgs = gen_imgs[0:3639, :, :]

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

# print('Calculate FID')
'''
gen_imgs_fid = gen_imgs.cpu().numpy()
gt_imgs_shuffle_fid = gen_imgs.cpu().numpy()
gen_imgs_fid = gen_imgs_fid.swapaxes(0, 2)
gt_imgs_shuffle_fid = gt_imgs_shuffle_fid.swapaxes(0, 2)
'''
'''
gen_imgs_avg = []
gt_imgs_shuffle_avg = []

for i in range (0, 10):
    gen_imgs_avg.append(gen_imgs[:,0,4800*i:4800*(i+1)])
    gt_imgs_shuffle_avg.append(gt_imgs[:,0,4800*i:4800*(i+1)])

gen_imgs_avg = np.mean(np.stack(gen_imgs_avg), axis=0)
gt_imgs_shuffle_avg = np.mean(np.stack(gt_imgs_shuffle_avg), axis=0)
'''
fid_value = eval_fid(gen_imgs, gt_imgs)

# print('Evaluation finished')

print('[RSME] MEAN : ' + str(rsme_mean) + ' / VAR : ' + str(rsme_var))
print('[PSNR] MEAN : ' + str(psnr_mean) + ' / VAR : ' + str(psnr_var))
print('[COS] MEAN : ' + str(fre_cos_mean) + ' / VAR : ' + str(fre_cos_var))
print('[MMD] MEAN : ' + str(mmd_mean) + ' / VAR : ' + str(mmd_var))
print('[EMD] MEAN : ' + str(emd_mean) + ' / VAR : ' + str(emd_var))
print('[FID] : ' + str(fid_value / 360.))
