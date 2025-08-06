import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask

import glob

class CustomDataset(Dataset):
    def __init__(
            self,
            name,
            data_root,
            window=64,
            proportion=0.8,
            save2npy=True,
            neg_one_to_one=True,
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
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata[0].shape[0], self.rawdata[0].shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.__len__()

    def __getsamples(self, data, proportion, seed):

        '''
        xs = []
        for idx in range(data.__len__()):
            x = np.zeros((self.sample_num_total, self.window, self.var_num))
            for i in range(self.sample_num_total):
                start = i
                end = i + self.window
                x[i, :, :] = data[idx][start:end, :]
            xs.append(x)
        sample_data = np.array(xs)
        print(sample_data.shape)
        exit(0)
        '''
        train_data, test_data = self.divide(data, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"),
                        self.unnormalize(test_data[0].copy()))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"),
                    self.unnormalize(train_data[0].copy()))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            unnormalize_to_zero_to_one(test_data[0].copy()))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        unnormalize_to_zero_to_one(train_data[0].copy()))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data[0])
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data[0])

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        # return d.reshape(-1, self.window, self.var_num)
        return d

    def __normalize(self, rawdatas):
        datas = []
        for rawdata in rawdatas:
            data = self.scaler.fit_transform(rawdata)
            if self.auto_norm:
                data = normalize_to_neg_one_to_one(data)
            datas.append(data)
        return datas

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.__len__()
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        '''
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]
        '''
        regular_data = data[:regular_train_num]
        irregular_data = data[regular_train_num:]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(data_root, name=''):

        datas = []
        file_list = glob.glob(data_root)
        scaler = MinMaxScaler()
        for filepath in file_list:
            data = np.load(filepath)
            scaler = scaler.fit(data)
            datas.append(data)

        return datas, scaler

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

    def __getitem__(self, ind, seed=2023):
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        sample_len = self.samples[0].shape[0]
        rand_num = np.random.randint(0, int(sample_len/self.window))

        if self.period == 'test':
            x = self.samples[ind]  # (seq_length, feat_dim) array
            m = self.masking[ind]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind][self.window*rand_num:self.window*(rand_num+1), :]  # (seq_length, feat_dim) array

        # Restore RNG.
        np.random.set_state(st0)

        return torch.from_numpy(x).float()

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
