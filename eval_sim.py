import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
from dtaidistance import dtw_visualisation
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 한글 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# 1. 비교할 두 개의 시계열 데이터
ts1 = np.load('/data/jionkim/diffusion_TS/OUTPUT/test_stock/ddpm_fake_test_stock.npy')
ts2 = np.load('/data/jionkim/diffusion_TS/OUTPUT/test_stock/samples/stock_norm_truth_24_train.npy')

ts1 = ts1[:3662, 0, 5]
ts2 = ts2[:3662, 0, 5]

dist_1_2 = np.linalg.norm(ts1 - ts2)
print(f"EUC 거리: {dist_1_2:.4f}")
'''
# 2. 코사인 유사도 계산
# scikit-learn은 2D 배열을 기대하므로 reshape이 필요합니다.
sim_1_2 = cosine_similarity(ts1, ts2)

# 2. DTW 거리 계산
# distance 변수에는 두 시계열 간의 DTW 거리가 저장됩니다.
# paths 변수에는 최적의 정렬 경로(warping path)가 저장됩니다.
'''
distance, path = fastdtw(ts1, ts2)
K = len(path)
if K == 0:
    normalized_distance = 0
else:
    normalized_distance = distance / K
print(f"DTW 거리: {normalized_distance:.4f}")

# 3. 경로 스텝별 거리 계산
path_distances = []
for i, j in path:
    # 매칭된 두 점 사이의 거리(절대값 차이)를 계산하여 리스트에 추가
    distance = abs(ts1[i] - ts2[j])
    path_distances.append(distance)

# 4. 시각화 (2개의 그래프를 위아래로 배치)
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False) # x축 공유 안 함

# 상단 그래프: 데이터시각화
axes[0].plot(ts1, label='Gen', color='blue', alpha=0.8)
axes[0].plot(ts2, label='GT', color='orange', alpha=0.8)
axes[0].set_title('Time Series Data Graph')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True)

# 중단 그래프: DTW 정렬 경로 시각화
axes[1].plot(ts1, label='Gen', color='blue', alpha=0.8)
axes[1].plot(ts2, label='GT', color='orange', alpha=0.8)
for i, j in path:
    axes[1].plot([i, j], [ts1[i], ts2[j]], color='gray', linestyle='--', linewidth=1)
axes[1].set_title('DTW Distance Visualization')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[1].grid(True)

# 하단 그래프: 경로 스텝별 거리 플롯
path_steps = np.arange(len(path_distances))
axes[2].plot(path_steps, path_distances, color='crimson', label='Local Cost')
# 거리가 큰 영역을 강조하기 위해 영역 채우기
axes[2].fill_between(path_steps, path_distances, color='crimson', alpha=0.2)
axes[2].set_title('DTW Path Distance Plot')
axes[2].set_xlabel('Warping Path Step')
axes[2].set_ylabel('Distance Between Nearest Points')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()