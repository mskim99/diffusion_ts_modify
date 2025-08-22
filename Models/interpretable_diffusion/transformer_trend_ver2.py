import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP, \
    AdaLayerNorm, Transpose, GELU2, series_decomp

class OutputAffine(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1,1,C))
        self.beta  = nn.Parameter(torch.zeros(1,1,C))
    def forward(self, x):  # x:(B,L,C)
        return self.gamma * x + self.beta

class SwiGLU(nn.Module):  # [UPD]
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class IndexConditionedEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.linear_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, class_index):
        emb = self.class_embedding(class_index)
        emb = self.linear_proj(emb)
        return emb


class TrendBlock(nn.Module):
    """
    Model trend of time series using a convolutional block.
    The polynomial regressor logic has been removed as it was incompatible with the model's architecture.
    """

    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        # This sequential module correctly processes a tensor of shape (B, C, T)
        # and returns a tensor of the same shape, making it suitable for residual connections.
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=trend_poly, out_channels=out_feat, kernel_size=3, stride=1, padding=1)
        )
        # The polynomial space logic is no longer used by the forward pass.
        # lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        # self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        # The input has shape (B, C, T). The self.trend module processes it
        # and returns a tensor of the same shape. This is the correct behavior.
        return self.trend(input)


class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """

    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """

    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 n_head,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 condition_embd,
                 n_head,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        # [UPD: 정확도] GELU→SwiGLU + LayerScale
        hidden = int(2/3 * mlp_hidden_times * n_embd)  # [UPD]
        self.mlp = nn.Sequential(                     # [UPD]
            SwiGLU(n_embd, hidden),
            nn.Dropout(resid_pdrop),
        )
        self.res_scale1 = nn.Parameter(torch.ones(1) * 1e-3)  # [UPD]
        self.res_scale2 = nn.Parameter(torch.ones(1) * 1e-3)  # [UPD]

    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + self.res_scale1 * a                 # [UPD: LayerScale]
        x = x + self.res_scale2 * self.mlp(self.ln2(x))  # [UPD: SwiGLU + LayerScale]
        return x, att


class Encoder(nn.Module):
    def __init__(
            self,
            n_layer=14,
            n_embd=1024,
            n_head=16,
            attn_pdrop=0.,
            resid_pdrop=0.,
            mlp_hidden_times=4,
            block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, index_emb, padding_masks=None, label_emb=None):
        x = input

        if index_emb is not None:
            x = x + index_emb.unsqueeze(1)

        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_channel,
                 n_feat,
                 for_trend,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        self.ln1_1 = AdaLayerNorm(n_embd)

        # [UPD: 정확도] GELU→SwiGLU + LayerScale
        hidden = int(2/3 * mlp_hidden_times * n_embd)   # [UPD]
        self.mlp = nn.Sequential(                       # [UPD]
            SwiGLU(n_embd, hidden),
            nn.Dropout(resid_pdrop),
        )
        self.linear = nn.Linear(n_embd, n_feat)
        self.res_scale1 = nn.Parameter(torch.ones(1) * 1e-3)  # [UPD]
        self.res_scale2 = nn.Parameter(torch.ones(1) * 1e-3)  # [UPD]
        self.res_scale3 = nn.Parameter(torch.ones(1) * 1e-3)  # [UPD] mainblock residual

        self.mainblock = TrendBlock(n_embd, n_embd, n_embd, n_embd, act=nn.GELU())

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None, index_emb=None):

        if index_emb is not None:
            x = x + index_emb.unsqueeze(1)

        # self-attn
        a, _ = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + self.res_scale1 * a  # [UPD: LayerScale]

        # cross-attn (label_emb 일관 주입)  # [UPD]
        a, _ = self.attn2(self.ln1_1(x, timestep, label_emb), encoder_output, mask=mask)  # [UPD]
        x = x + self.res_scale2 * a  # [UPD: LayerScale]

        # main block (trend/fourier) + residual  # [UPD]
        # h = self.mainblock(x)
        x_transposed = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        h_transposed = self.mainblock(x_transposed)
        h = h_transposed.transpose(1, 2)  # Restore as (B, T, C)

        x = x + self.res_scale3 * h  # [UPD: residual on main block]

        x = self.mlp(self.ln2(x))  # [UPD: SwiGLU]

        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m)


class Decoder(nn.Module):
    def __init__(
            self,
            n_channel,
            n_feat,
            for_trend,
            n_embd=1024,
            n_head=16,
            n_layer=10,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
            condition_dim=512
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(*[DecoderBlock(
            n_feat=n_feat,
            n_channel=n_channel,
            for_trend=for_trend,
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            condition_dim=condition_dim,
        ) for _ in range(n_layer)])

    def forward(self, x, t, enc, padding_masks=None, label_emb=None, index_emb=None):
        b, c, _ = x.shape
        mean = []
        for block_idx in range(len(self.blocks)):
            x, residual_mean = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb, index_emb=index_emb)
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean


class TransformerT(nn.Module):
    def __init__(
            self,
            n_feat,
            n_channel,
            input_length,
            for_trend,
            n_layer_enc=5,
            n_layer_dec=14,
            n_embd=1024,
            n_heads=16,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            mlp_hidden_times=4,
            block_activate='GELU',
            max_len=2048,
            conv_params=None,
            **kwargs
    ):
        super().__init__()

        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)
        self.out_affine = OutputAffine(n_feat)

        self.idx_emb = IndexConditionedEmbedding(int(input_length // n_channel) + 1, n_embd)

        self.combine = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                 padding_mode='circular', bias=False)
        nn.init.xavier_uniform_(self.combine.weight)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.final_encoder_norm = nn.LayerNorm(n_embd)

        self.decoder = Decoder(n_channel, n_feat, for_trend, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop,
                               mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.layer_weights = nn.Parameter(torch.zeros(n_layer_dec))  # [UPD]

    def forward(self, input, t, index=None, padding_masks=None, return_res=False):

        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)

        if index is not None:
            index_emb = self.idx_emb(index)  # shape: [B, D]
        else:
            index_emb = None

        enc_cond = self.encoder(inp_enc, t, index_emb=index_emb, padding_masks=padding_masks)
        enc_cond = self.final_encoder_norm(enc_cond)

        inp_dec = self.pos_dec(emb)
        output, mean = self.decoder(inp_dec, t, enc_cond, index_emb=index_emb, padding_masks=padding_masks)

        res = self.inverse(output)

        # mean: [B, n_layer_dec, 1, D_emb]
        # w = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)  # [1,L,1,1]
        # combined_mean = (mean * w).sum(dim=1, keepdim=True)            # [B,1,1,D_emb]
        # res = combined_mean + res

        # res = torch.tanh(res)
        res = self.out_affine(res)

        return res