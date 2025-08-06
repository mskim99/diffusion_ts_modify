import numpy as np
import os
import glob
import random

from Utils.eval_utils import rmse,psnr,fre_cosine,mmd,emd, eval_fid
from Utils.metric_utils import visualization

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
gen_imgs = np.load('/data/jionkim/diffusion_TS/OUTPUT/test_stocks_multi_w_3662/samples/dpm_fake_test_textum_multi_w_1024.npy')
# gt_imgs = np.load('/data/jionkim/diffusion_TS/OUTPUT/test_textum_multi_w_1024/samples/textum_norm_truth_1024_train.npy')
gt_imgs_path = glob.glob('/data/jionkim/diffusion_TS/OUTPUT/test_stocks_multi_w_3662/samples/*.npy')
random.shuffle(gt_imgs_path)

gt_imgs = []
for i in range (0, 16):
    print(gt_imgs_path[i])
    gt_img = np.load(gt_imgs_path[i])
    gt_imgs.append(gt_img)
gt_imgs = np.array(gt_imgs)

gen_imgs = gen_imgs.reshape([16, -1, 5])

'''
mean = np.mean(gen_imgs)
std = np.std(gen_imgs)
# gen_imgs = (gen_imgs - mean) / std

mean = np.mean(gt_imgs)
std = np.std(gt_imgs)
# gt_imgs = (gt_imgs - mean) / std
'''

gen_imgs = gen_imgs[:, 0:48000, :]
gt_imgs = gt_imgs[0:48000, :]

# gen_imgs = np.reshape(gen_imgs, (48000, -1))
# gt_imgs = np.reshape(gt_imgs, (48000, -1))

# gen_imgs = np.swapaxes(gen_imgs, 1, 0)
# gt_imgs = np.swapaxes(gt_imgs, 1, 0)

print(gen_imgs.shape)
print(gt_imgs.shape)

# gen_imgs = np.expand_dims(gen_imgs, 1)
# gt_imgs = np.expand_dims(gt_imgs, 1)

# gen_imgs = np.concatenate([gen_imgs, gen_imgs], axis=1)
# gt_imgs = np.concatenate([gt_imgs, gt_imgs], axis=1)

gen_imgs = np.swapaxes(gen_imgs, 1, 0)
gt_imgs = np.swapaxes(gt_imgs, 1, 0)

print(gen_imgs.shape)
print(gt_imgs.shape)

visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='pca', compare=gt_imgs.shape[0])
visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='tsne', compare=gt_imgs.shape[0])
visualization(ori_data=gt_imgs, generated_data=gen_imgs, analysis='kernel', compare=gt_imgs.shape[0])

# gen_imgs = torch.Tensor(gen_imgs).cuda()
# gt_imgs_shuffle = torch.Tensor(gt_imgs_shuffle).cuda()

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
