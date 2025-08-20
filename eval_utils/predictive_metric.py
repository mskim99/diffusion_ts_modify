"""Reimplement TimeGAN-pytorch Codebase.

predictive_metrics.py
Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
# sys.path.append('/sciclone/home/...')

import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf1
tf.compat.v1.disable_eager_execution()
import numpy as np
from sklearn.metrics import mean_absolute_error
from .metric_utils import extract_time


# -----------------------------
# [FIX] helper: build padded batches (fixed shapes)
# -----------------------------
def _make_padded_batch(data, time_lens, idx, max_seq_len, dim):
    """
    data: list/array of sequences, each shape (Ti, dim)
    time_lens: list/array of lengths Ti
    idx: indices to select
    returns:
      X: (B, max_seq_len-1, dim-1)
      Y: (B, max_seq_len-1, 1)
      T: (B,)
    """
    B = len(idx)
    L = max_seq_len - 1
    D_in = dim - 1

    X = np.zeros((B, L, D_in), dtype=np.float32)
    Y = np.zeros((B, L, 1),     dtype=np.float32)
    T = np.zeros((B,),          dtype=np.int32)

    for j, i in enumerate(idx):
        t = int(time_lens[i])
        eff = max(1, min(t - 1, L))
        seq = np.asarray(data[i], dtype=np.float32)
        Xi = seq[:eff, :D_in]                  # (eff, dim-1)
        Yi = seq[1:eff + 1, D_in]              # (eff,)
        X[j, :eff, :] = Xi
        Y[j, :eff, 0] = Yi
        T[j] = eff
    return X, T, Y


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
    Args:
      - ori_data: original data (N, T, D)
      - generated_data: generated synthetic data (N, T, D)
    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    # Initialization on the Graph
    tf1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = ori_data.shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Builde a post-hoc RNN predictive network
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 16

    # Input place holders (fixed time length = max_seq_len-1)
    X = tf1.placeholder(tf.float32, [None, max_seq_len - 1, dim - 1], name="myinput_x")
    T = tf1.placeholder(tf.int32,   [None],                           name="myinput_t")
    Y = tf1.placeholder(tf.float32, [None, max_seq_len - 1, 1],       name="myinput_y")

    # Predictor function
    def predictor(x, t):
        with tf1.variable_scope("predictor", reuse=tf1.AUTO_REUSE) as vs:
            p_cell = tf1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='p_cell')
            p_outputs, p_last_states = tf1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length=t)
            y_hat_logit = tf1.layers.dense(p_outputs, 1, activation=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            p_vars = [v for v in tf1.global_variables() if v.name.startswith(vs.name)]
        return y_hat, p_vars

    y_pred, p_vars = predictor(X, T)

    # Loss & optimizer
    p_loss = tf1.losses.absolute_difference(Y, y_pred)
    p_solver = tf1.train.AdamOptimizer().minimize(p_loss, var_list=p_vars)

    # -----------------------------
    # [FIX] TF1 session: disable GPU + allow_growth
    # -----------------------------
    config = tf1.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True     # allocate as needed
    config.device_count['GPU'] = 0             # force CPU for this metric
    sess = tf1.Session(config=config)
    sess.run(tf1.global_variables_initializer())

    from tqdm import tqdm

    # -----------------------------
    # Training using Synthetic dataset (padded np batches)
    # -----------------------------
    gen_N = len(generated_data)
    for _ in tqdm(range(iterations), desc='training', total=iterations):
        idx = np.random.permutation(gen_N)[:batch_size]
        X_mb, T_mb, Y_mb = _make_padded_batch(
            generated_data, generated_time, idx, max_seq_len, dim
        )
        # Train predictor
        _, _loss = sess.run([p_solver, p_loss],
                            feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

    # -----------------------------
    # [FIX] Batched inference on original data (avoid peak memory)
    # -----------------------------
    def _batched_predict(data, time_lens, bs=128):
        N = len(data)
        preds = []
        for s in range(0, N, bs):
            e = min(N, s + bs)
            idx = np.arange(s, e)
            X_b, T_b, _ = _make_padded_batch(data, time_lens, idx, max_seq_len, dim)
            y_b = sess.run(y_pred, feed_dict={X: X_b, T: T_b})
            preds.append(y_b)
        return np.concatenate(preds, axis=0)

    pred_Y_curr = _batched_predict(ori_data, ori_time, bs=128)

    # -----------------------------
    # MAE on original data (padded GT batch to match shapes)
    # -----------------------------
    X_full, T_full, Y_full = _make_padded_batch(
        ori_data, ori_time, np.arange(no), max_seq_len, dim
    )

    # MAE: 각 샘플 t_i 유효 길이만 사용
    total = 0.0
    for i in range(no):
        L = int(T_full[i])
        if L <= 0:
            continue
        total += mean_absolute_error(Y_full[i, :L, :], pred_Y_curr[i, :L, :])
    predictive_score = total / no

    return predictive_score
