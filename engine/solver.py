import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
import random

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model_trend, model_season, dataloader, logger=None):
        super().__init__()
        self.model_trend = model_trend
        self.model_season = model_season
        self.device = self.model_trend.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model_trend.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt_tr = Adam(filter(lambda p: p.requires_grad, self.model_trend.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema_tr = EMA(self.model_trend, beta=ema_decay, update_every=ema_update_every).to(self.device)

        self.opt_se = Adam(filter(lambda p: p.requires_grad, self.model_season.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema_se = EMA(self.model_season, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg_tr = config['solver']['scheduler']
        sc_cfg_tr['params']['optimizer'] = self.opt_tr
        self.sch_tr = instantiate_from_config(sc_cfg_tr)

        sc_cfg_se = config['solver']['scheduler']
        sc_cfg_se['params']['optimizer'] = self.opt_se
        self.sch_se = instantiate_from_config(sc_cfg_se)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model_trend)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model_trend': self.model_trend.state_dict(),
            'ema_trend': self.ema_tr.state_dict(),
            'opt_trend': self.opt_tr.state_dict(),
            'model_season': self.model_season.state_dict(),
            'ema_season': self.ema_se.state_dict(),
            'opt_season': self.opt_se.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def save_finetune(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'ckpt_ft-{milestone}.pt')))
        data = {
            'step': self.step,
            'model_trend': self.model_trend.state_dict(),
            'ema_trend': self.ema_tr.state_dict(),
            'opt_trend': self.opt_tr.state_dict(),
            'model_season': self.model_season.state_dict(),
            'ema_season': self.ema_se.state_dict(),
            'opt_season': self.opt_se.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'ckpt_ft-{milestone}.pt'))

    def save_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,
            'classifier': self.classifier.state_dict()
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    def load(self, milestone, verbose=False, finetune=False):
        if self.logger is not None and verbose:
            if finetune:
                self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_ft-{milestone}.pt')))
            else:
                self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        if finetune:
            data = torch.load(str(self.results_folder / f'ckpt_ft-{milestone}.pt'), map_location=device)
        else:
            data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model_trend.load_state_dict(data['model_trend'])
        self.model_season.load_state_dict(data['model_season'])
        self.step = data['step']
        self.opt_tr.load_state_dict(data['opt_trend'])
        self.ema_tr.load_state_dict(data['ema_trend'])
        self.opt_se.load_state_dict(data['opt_season'])
        self.ema_se.load_state_dict(data['ema_season'])
        self.milestone = milestone

    def load_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])
        self.step_classifier = data['step']
        self.milestone_classifier = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                total_loss_tr = 0.
                total_loss_se = 0.
                total_loss_f = 0.
                total_loss_d = 0.
                for _ in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)
                    data = next(self.dl)
                    data_trend = data[0].to(device)
                    data_season = data[1].to(device)

                    data_trend = (data_trend - data_trend.min()) / (data_trend.max() - data_trend.min())
                    data_trend = 2. * data_trend - 1.
                    data_season = (data_season - data_season.min()) / (data_season.max() - data_season.min())
                    data_season = 2. * data_season - 1.
                    data_idx = data[2].to(device)

                    # data = torch.swapaxes(data, 0, 1) # for textum
                    loss_tr, _, _ = self.model_trend(data_trend, index=None, target=data_trend)
                    loss_se, loss_f, _ = self.model_season(data_season, index=None, target=data_season)
                    loss_tr = loss_tr / self.gradient_accumulate_every
                    loss_se = loss_se / self.gradient_accumulate_every
                    (loss_tr + loss_se).backward()
                    total_loss_tr += loss_tr.item()
                    total_loss_se += loss_se.item()
                    total_loss = total_loss_tr + total_loss_se
                    total_loss_f += loss_f.item()
                    # total_loss_d += loss_d.item()

                pbar.set_description(f'loss/ tr: {total_loss_tr:.4f}, se: {total_loss_se:.4f}, fourier: {total_loss_f:.4f}')

                clip_grad_norm_(self.model_trend.parameters(), 1.0)
                self.opt_tr.step()
                self.sch_tr.step(total_loss_tr)
                self.opt_tr.zero_grad()

                self.opt_se.step()
                self.sch_se.step(total_loss_se)
                self.opt_se.zero_grad()

                self.ema_tr.update()
                self.ema_se.update()

                self.step += 1
                step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def train_sap(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                total_loss_tr = 0.
                total_loss_se = 0.
                total_loss_f = 0.
                # total_loss_d = 0.

                # Gradient Accumulation Loop
                # for _ in range(self.gradient_accumulate_every):
                for data in self.dataloader:
                    # data = next(self.dl)
                    data_trend = data[0].to(device)
                    data_season = data[1].to(device)
                    data_idx = data[2].to(device)

                    # Data Preprocessing
                    data_trend = (data_trend - data_trend.min()) / (data_trend.max() - data_trend.min())
                    data_trend = 2. * data_trend - 1.
                    data_season = (data_season - data_season.min()) / (data_season.max() - data_season.min())
                    data_season = 2. * data_season - 1.

                    loss_tr, _, _ = self.model_trend(data_trend, index=None, target=data_trend)
                    loss_tr.backward()
                    total_loss_tr += loss_tr.item()

                    # --- 2. Season 모델 독립적 훈련 ---
                    loss_se, loss_f, _ = self.model_season(data_season, index=None, target=data_season)
                    loss_se.backward()
                    total_loss_se += loss_se.item()

                    # 로깅을 위한 손실 집계
                    total_loss = total_loss_tr + total_loss_se
                    # loss_f는 로깅용으로 보이며, 원본 코드처럼 gradient accumulation 스케일링 없이 합산
                    total_loss_f += loss_f.item()
                    # total_loss_d += loss_d.item()

                pbar.set_description(
                    f'loss/ tr: {total_loss_tr:.4f}, se: {total_loss_se:.4f}, fourier: {total_loss_f:.4f}')

                # --- 3. 각 모델의 파라미터 업데이트 ---
                # Trend 모델 업데이트
                clip_grad_norm_(self.model_trend.parameters(), 1.0)
                self.opt_tr.step()
                self.sch_tr.step(total_loss_tr)
                self.opt_tr.zero_grad()

                # Season 모델 업데이트 (안정적인 훈련을 위해 clip_grad_norm_ 추가)
                clip_grad_norm_(self.model_season.parameters(), 1.0)
                self.opt_se.step()
                self.sch_se.step(total_loss_se)
                self.opt_se.zero_grad()

                self.ema_tr.update()
                self.ema_se.update()

                self.step += 1
                step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def train_finetune(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.

                # Gradient Accumulation Loop
                # for _ in range(self.gradient_accumulate_every):
                for data in self.dataloader:
                    # data = next(self.dl)
                    data_trend = data[0].to(device)
                    data_season = data[1].to(device)
                    data_idx = data[2].to(device)

                    data_recon = data_trend + data_season

                    dt_min = data_trend.min()
                    dt_max = data_trend.max()
                    ds_min = data_season.min()
                    ds_max = data_season.max()

                    # Data Preprocessing
                    data_trend_norm = (data_trend - dt_min) / (dt_max - dt_min)
                    data_trend_norm = 2. * data_trend_norm - 1.
                    data_season_norm = (data_season - ds_min) / (ds_max - ds_min)
                    data_season_norm = 2. * data_season_norm - 1.

                    result_trend = self.model_trend(data_trend_norm, index=None, target=data_trend_norm, out_result=True)
                    result_season = self.model_season(data_season_norm, index=None, target=data_season_norm, out_result=True)
                    result_trend = (result_trend - result_trend.min()) / (result_trend.max() - result_trend.min()) * \
                                   (dt_max - dt_min) + dt_min
                    result_season = (result_season - result_season.min()) / (result_season.max() - result_season.min()) * \
                                   (ds_max - ds_min) + ds_min
                    result_recon = result_trend + result_season
                    loss_recon = F.l1_loss(result_recon, data_recon)

                    loss_recon.backward()
                    total_loss += loss_recon.item()

                pbar.set_description(
                    f'loss_ft: {total_loss:.4f}')

                # --- 3. 각 모델의 파라미터 업데이트 ---
                # Trend 모델 업데이트
                clip_grad_norm_(self.model_trend.parameters(), 1.0)
                self.opt_tr.step()
                self.sch_tr.step(total_loss)
                self.opt_tr.zero_grad()

                # Season 모델 업데이트 (안정적인 훈련을 위해 clip_grad_norm_ 추가)
                clip_grad_norm_(self.model_season.parameters(), 1.0)
                self.opt_se.step()
                self.sch_se.step(total_loss)
                self.opt_se.zero_grad()

                self.ema_tr.update()
                self.ema_se.update()

                self.step += 1
                step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save_finetune(self.milestone)

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train_ft/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None, norm_factor=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        trends = np.empty([0, shape[0], shape[1]])
        seasons = np.empty([0, shape[0], shape[1]])
        # num_cycle = int(num // size_every) + 1
        bs_num = 200

        print('num_cycle: ' + str(num))
        for _ in range(num):
            trend = self.ema_tr.ema_model.generate_mts(batch_size=bs_num, model_kwargs=model_kwargs, cond_fn=cond_fn)
            season = self.ema_se.ema_model.generate_mts(batch_size=bs_num, model_kwargs=model_kwargs, cond_fn=cond_fn)
            trends = np.row_stack([trends, trend.detach().cpu().numpy()])
            seasons = np.row_stack([seasons, season.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if norm_factor is not None:
            norm_idx = np.array([random.randint(0, norm_factor[0].__len__()-1) for _ in range(bs_num * num)])
            t_min = np.array(norm_factor[0])[norm_idx]
            t_max = np.array(norm_factor[1])[norm_idx]
            s_min = np.array(norm_factor[2])[norm_idx]
            s_max = np.array(norm_factor[3])[norm_idx]

            trends = (trends - trends.min()) / (trends.max() - trends.min()) * (t_max - t_min) + t_min
            seasons = (seasons - seasons.min()) / (seasons.max() - seasons.min()) * (s_max - s_min) + s_min

        samples = trends + seasons

        '''
        for _ in range(num):
            sample, trend, season = self.ema.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            trends = np.row_stack([trends, trend.detach().cpu().numpy()])
            seasons = np.row_stack([seasons, season.detach().cpu().numpy()])
            torch.cuda.empty_cache()
        '''

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))

        return samples, trends, seasons

    def sample_idx_emb(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        device = self.device
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        trends = np.empty([0, shape[0], shape[1]])
        seasons = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        print('num_cycle: ' + str(num_cycle))
        for idx in range(num_cycle):
            idx_input = torch.Tensor([idx]).to(device).long()
            trend = self.ema_tr.ema_model.generate_mts(batch_size=1, idx=idx_input, model_kwargs=model_kwargs, cond_fn=cond_fn)
            season = self.ema_se.ema_model.generate_mts(batch_size=1, idx=idx_input, model_kwargs=model_kwargs, cond_fn=cond_fn)
            trends = np.row_stack([trends, trend.detach().cpu().numpy()])
            seasons = np.row_stack([seasons, season.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        samples = trends + seasons

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))

        return samples, trends, seasons

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    logits = classifier(x_t, t)
                    loss = F.cross_entropy(logits, y)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                self.opt_classifier.step()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier

