import os
import torch
import numpy as np
import random

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask

from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        trend_data,
        season_data,
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=False,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.trend_data, self.season_data = self.read_data(trend_data, season_data, self.name)
        # self.t_min, self.t_max = self.trend_data.min(), self.trend_data.max()
        # self.s_min, self.s_max = self.season_data.min(), self.season_data.max()
        # self.t_min, self.t_max = [], []
        # self.s_min, self.s_max = [], []
        self.inv_cdfs_trend = []
        self.inv_cdfs_season = []
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.trend_data.shape[0], self.trend_data.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        # self.sample_num_total = max(self.len // self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        train_trend, inference_trend, train_season, inference_season \
            = self.__getsamples(self.trend_data, self.season_data, proportion, seed)

        for i in range(self.trend_data.shape[1]):
            inv_cdf_trend = self.kde_cdf_inverse(self.trend_data[:, i], bw_method=0.3)
            inv_cdf_season = self.kde_cdf_inverse(self.season_data[:, i], bw_method=0.3)
            self.inv_cdfs_trend.append(inv_cdf_trend)
            self.inv_cdfs_season.append(inv_cdf_season)

        # np.savez(os.path.join(self.dir, f"scaler_ts.npz"), t_min=self.t_min, t_max=self.t_max, s_min=self.s_min, s_max=self.s_max)

        self.samples_trend = train_trend if period == 'train' else inference_trend
        self.samples_season = train_season if period == 'train' else inference_season
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples_trend.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

        self.sample_num = self.samples_trend.shape[0]

    def __getsamples(self, trend_data, season_data, proportion, seed):
        trend = np.zeros((self.sample_num_total, self.window, self.var_num))
        season = np.zeros((self.sample_num_total, self.window, self.var_num))

        for i in range(self.sample_num_total):
            # start = i * self.window
            # end = min(((i + 1) * self.window), self.len)
            start = i
            end = i + self.window
            trend[i, 0:(end-start), :] = trend_data[start:end, :]
            season[i, 0:(end-start), :] = season_data[start:end, :]
            '''
            self.t_min.append(trend[i, :, :].min(axis=0))
            self.t_max.append(trend[i, :, :].max(axis=0))
            self.s_min.append(season[i, :, :].min(axis=0))
            self.s_max.append(season[i, :, :].max(axis=0))
            '''

        train_data_trend, test_data_trend, train_data_season, test_data_season = self.divide(trend, season, proportion, seed)

        if self.save2npy:
            '''
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data_trend+test_data_season))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data_trend+train_data_season))
            '''
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data_trend+test_data_season))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data_trend+train_data_season))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data_trend+test_data_season)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train_trend.npy"), train_data_trend)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train_season.npy"), train_data_season)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data_trend+train_data_season)

        '''
        if self.period == 'train':
            aug_trend_list = []
            aug_season_list = []
            for i in range(30):
                aug_trend = self.time_series_augmentation(train_data_trend, methods='jitter')
                aug_season = self.time_series_augmentation(train_data_season, methods='jitter')
                aug_trend_list.append(aug_trend)
                aug_season_list.append(aug_season)

            aug_trend_array = np.vstack(aug_trend_list)
            aug_season_array = np.vstack(aug_season_list)

            train_data_trend = np.concatenate([train_data_trend, aug_trend_array], axis=0)
            train_data_season = np.concatenate([train_data_season, aug_season_array], axis=0)
        '''
        return train_data_trend, test_data_trend, train_data_season, test_data_season


    def kde_cdf_inverse(self, data, bw_method=0.3, grid_size=1000, margin=0.05):

        data = np.asarray(data).ravel()
        x_min, x_max = data.min(), data.max()
        span = x_max - x_min if x_max > x_min else 1.0
        x_grid = np.linspace(x_min - margin * span, x_max + margin * span, grid_size)

        kde = gaussian_kde(data, bw_method=bw_method)
        density = kde(x_grid)

        # CDF 계산
        cdf = cumulative_trapezoid(density, x_grid, initial=0.0)
        cdf /= cdf[-1] if cdf[-1] != 0 else 1.0

        # 역함수 보간
        inverse_cdf_func = interp1d(cdf, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))
        return inverse_cdf_func


    def time_series_augmentation(self, sample, methods=('jitter', 'scaling', 'permutation'), jitter_std=0.05,
                                 scale_range=(0.9, 1.1)):
        """Augment a single time series sample (window x features)"""
        augmented = sample.copy()
        method = random.choice(methods)

        if method == 'jitter':
            noise = np.random.normal(loc=0.0, scale=jitter_std, size=augmented.shape)
            augmented += noise

        elif method == 'scaling':
            factor = np.random.uniform(scale_range[0], scale_range[1])
            augmented *= factor

        elif method == 'permutation':
            num_splits = np.random.randint(2, 5)
            splits = np.array_split(augmented, num_splits, axis=0)
            np.random.shuffle(splits)
            augmented = np.concatenate(splits, axis=0)

        return augmented

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(trend_data, season_data, ratio, seed=2023):
        size = trend_data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data_trend = trend_data[regular_train_id, :]
        irregular_data_trend = trend_data[irregular_train_id, :]
        regular_data_season = season_data[regular_train_id, :]
        irregular_data_season = season_data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data_trend, irregular_data_trend, regular_data_season, irregular_data_season

    @staticmethod
    def read_data(trend_file, season_file, name=''):
        """Reads a single .csv
        """
        trend = np.load(trend_file)
        season = np.load(season_file)
        return trend, season
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x_tr = self.samples_trend[ind, :, :]  # (seq_length, feat_dim) array
            x_se = self.samples_season[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x_tr).float(), torch.from_numpy(x_se).float(), torch.from_numpy(m)
        x_tr = self.samples_trend[ind, :, :]  # (seq_length, feat_dim) array
        x_se = self.samples_season[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x_tr).float(), torch.from_numpy(x_se).float(), ind

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
