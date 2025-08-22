import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
from torch import nn   # 추가

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

        # -------------------------
        # [FUSION] 학습 가능한 게이트 α
        # -------------------------
        C = model_trend.feature_size   # or channel dimension
        self.fusion_alpha = nn.Parameter(torch.zeros(1, 1, C, device=self.device))
        self.opt_fuse = Adam([self.fusion_alpha], lr=start_lr, betas=[0.9, 0.96])

    # -------------------------
    # 저장 / 로드
    # -------------------------
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
            # [FUSION]
            'fusion_alpha': self.fusion_alpha.detach().cpu(),
            'opt_fuse': self.opt_fuse.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model_trend.load_state_dict(data['model_trend'])
        self.model_season.load_state_dict(data['model_season'])
        self.step = data['step']
        self.opt_tr.load_state_dict(data['opt_trend'])
        self.ema_tr.load_state_dict(data['ema_trend'])
        self.opt_se.load_state_dict(data['opt_season'])
        self.ema_se.load_state_dict(data['ema_season'])
        self.milestone = milestone
        # [FUSION]
        if 'fusion_alpha' in data:
            self.fusion_alpha = nn.Parameter(data['fusion_alpha'].to(device))
        if 'opt_fuse' in data:
            self.opt_fuse = Adam([self.fusion_alpha], lr=self.config['solver'].get('base_lr', 1.0e-4))
            self.opt_fuse.load_state_dict(data['opt_fuse'])

    # -------------------------
    # 학습 루틴
    # -------------------------
    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss_tr, total_loss_se, total_loss_fuse = 0., 0., 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data_trend = data[0].to(device)
                    data_season = data[1].to(device)
                    x_gt = data_trend + data_season  # 전체 GT 시계열
                    xmin, xmax = x_gt.min(), x_gt.max()
                    def norm(z): return 2. * (z - xmin) / (xmax - xmin + 1e-6) - 1.

                    data_trend_n = norm(data_trend)
                    data_season_n = norm(data_season)
                    x_gt_n = norm(x_gt)

                    # 모델 출력
                    loss_tr, _, _, trend_out = self.model_trend(data_trend_n, index=None, target=data_trend_n, out_result=True)
                    loss_se, _, _, season_out = self.model_season(data_season_n, index=None, target=data_season_n, out_result=True)

                    # Fusion 출력 및 loss
                    alpha = torch.sigmoid(self.fusion_alpha)  # (1,1,C)
                    fused_out = alpha * trend_out + (1 - alpha) * season_out
                    loss_fuse = F.mse_loss(fused_out, x_gt_n)

                    # 총 손실
                    total_loss = loss_tr + loss_se + loss_fuse
                    (total_loss / self.gradient_accumulate_every).backward()

                    total_loss_tr += loss_tr.item()
                    total_loss_se += loss_se.item()
                    total_loss_fuse += loss_fuse.item()

                pbar.set_description(
                    f'loss/ tr: {total_loss_tr:.4f}, se: {total_loss_se:.4f}, fuse: {total_loss_fuse:.4f}'
                )

                # 옵티마 업데이트
                clip_grad_norm_(self.model_trend.parameters(), 1.0)
                self.opt_tr.step()
                self.sch_tr.step(total_loss_tr)
                self.opt_tr.zero_grad()

                clip_grad_norm_(self.model_season.parameters(), 1.0)
                self.opt_se.step()
                self.sch_se.step(total_loss_se)
                self.opt_se.zero_grad()

                self.opt_fuse.step()
                self.opt_fuse.zero_grad()  # α 업데이트

                # EMA 업데이트
                self.ema_tr.update()
                self.ema_se.update()

                self.step += 1
                step += 1
                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss',
                                               scalar_value=total_loss_tr + total_loss_se + total_loss_fuse,
                                               global_step=self.step)
                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    # -------------------------
    # [FUSION] 적용
    # -------------------------
    def _apply_fusion(self, trends, seasons):
        fa = torch.sigmoid(self.fusion_alpha).detach().cpu().numpy()  # (1,1,C)
        return fa * trends + (1.0 - fa) * seasons

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        if self.logger is not None: tic = time.time(); self.logger.log_info('Begin to sample...')
        trends, seasons = np.empty([0, shape[0], shape[1]]), np.empty([0, shape[0], shape[1]])
        bs_num = 40
        for _ in range(num):
            season = self.ema_se.ema_model.generate_mts(batch_size=bs_num, model_kwargs=model_kwargs, cond_fn=cond_fn)
            trend  = self.ema_tr.ema_model.generate_mts(batch_size=bs_num, model_kwargs=model_kwargs, cond_fn=cond_fn)
            trends = np.row_stack([trends, trend.detach().cpu().numpy()])
            seasons= np.row_stack([seasons,season.detach().cpu().numpy()])
            torch.cuda.empty_cache()
        samples = self._apply_fusion(trends, seasons)
        if self.logger is not None: self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time()-tic))
        return samples, trends, seasons
