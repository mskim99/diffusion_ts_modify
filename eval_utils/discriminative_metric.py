"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
## Replace this line to register eval_utils in improved_diffusion/eval_utils
sys.path.append('/sciclone/home/yli102/diffusion_rebuild/improved_diffusion/eval_utils')
import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf1
import numpy as np
from sklearn.metrics import accuracy_score
from .metric_utils import train_test_divide, extract_time


def batch_generator(data, time, batch_size):
  """Mini-batch generator.

  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch

  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]

  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)

  return X_mb, T_mb


def discriminative_score_metrics (ori_data, generated_data):
  """Use post-hoc RNN to classify original data and synthetic data"""
  # ------------------------------------------
  # [D] 추가: 같은 프로세스에서 PyTorch가 GPU를 점유하고 있으면
  #           TF의 allow_growth 효과가 체감되지 않습니다.
  #           지표 계산 직전에 캐시를 비워 GPU 점유를 해제합니다.
  try:
    import torch
    torch.cuda.empty_cache()   # [D]
  except Exception:
    pass
  # ------------------------------------------

  tf1.reset_default_graph()

  no, seq_len, dim = np.asarray(ori_data).shape
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

  hidden_dim = int(dim/2)
  iterations = 2000
  batch_size = 16

  # ------------------------------------------
  # [C] 추가: 판별기 평가 그래프를 전부 CPU에 올립니다.
  #           (평가용이라 속도 영향 미미, GPU 사용량은 즉시 0에 수렴)
  with tf1.device('/CPU:0'):  # [C]
    X = tf1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
    X_hat = tf1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")
    T = tf1.placeholder(tf.int32, [None], name = "myinput_t")
    T_hat = tf1.placeholder(tf.int32, [None], name = "myinput_t_hat")

    def discriminator (x, t):
      with tf1.variable_scope("discriminator", reuse = tf1.AUTO_REUSE) as vs:
        d_cell = tf1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='d_cell')
        d_outputs, d_last_states = tf1.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length=t)
        y_hat_logit = tf1.layers.dense(d_last_states, 1, activation=None)
        y_hat = tf.nn.sigmoid(y_hat_logit)
        d_vars = [v for v in tf1.all_variables() if v.name.startswith(vs.name)]
      return y_hat_logit, y_hat, d_vars

    y_logit_real, y_pred_real, d_vars = discriminator(X, T)
    y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)

    d_loss_real = tf1.reduce_mean(
        tf1.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_real, labels=tf1.ones_like(y_logit_real)))
    d_loss_fake = tf1.reduce_mean(
        tf1.nn.sigmoid_cross_entropy_with_logits(logits=y_logit_fake, labels=tf1.zeros_like(y_logit_fake)))
    d_loss = d_loss_real + d_loss_fake

    # (B 적용 버전 그대로 두어도 되고, 단일 옵티마이저로 간소화해도 됩니다.)
    d_solver_real = tf1.train.AdamOptimizer().minimize(d_loss_real, var_list=d_vars)   # [B 기존 유지]
    d_solver_fake = tf1.train.AdamOptimizer().minimize(d_loss_fake, var_list=d_vars)   # [B 기존 유지]
  # ------------------------------------------

  # ------------------------------------------
  # [A + C] 세션도 GPU를 완전히 비활성화합니다.
  #         allow_growth는 남겨두되, device_count로 GPU=0을 명시.
  config = tf1.ConfigProto()
  config.gpu_options.allow_growth = True   # [A]
  config.allow_soft_placement = True
  config.device_count['GPU'] = 0           # [C] 핵심: 이 세션은 GPU를 안 씀
  sess = tf1.Session(config=config)
  # ------------------------------------------

  sess.run(tf1.global_variables_initializer())

  train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
      train_test_divide(ori_data, generated_data, ori_time, generated_time)

  from tqdm import tqdm
  for itt in tqdm(range(iterations), desc='training', total=iterations):
    X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
    X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

    # [B] 교대 업데이트 (CPU 상에서 수행됨)
    _ , step_d_loss_real = sess.run(
        [d_solver_real, d_loss_real],
        feed_dict={X: X_mb, T: T_mb, X_hat: X_mb, T_hat: T_mb}
    )
    _ , step_d_loss_fake = sess.run(
        [d_solver_fake, d_loss_fake],
        feed_dict={X: X_hat_mb, T: T_hat_mb, X_hat: X_hat_mb, T_hat: T_hat_mb}
    )

  y_pred_real_curr, y_pred_fake_curr = sess.run(
      [y_pred_real, y_pred_fake],
      feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat}
  )

  y_pred_final  = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis=0)
  acc      = accuracy_score(y_label_final, (y_pred_final > 0.5))
  fake_acc = accuracy_score(np.zeros([len(y_pred_fake_curr),]), (y_pred_fake_curr > 0.5))
  real_acc = accuracy_score(np.ones([len(y_pred_fake_curr),]),  (y_pred_real_curr > 0.5))
  discriminative_score = np.abs(0.5 - acc)
  return discriminative_score, fake_acc, real_acc, (y_pred_fake_curr>0.5, y_pred_real_curr>0.5, test_x_hat, test_x)