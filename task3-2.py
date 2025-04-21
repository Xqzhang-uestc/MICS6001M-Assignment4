import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.table import Table
from itertools import product
from matplotlib.patches import FancyArrowPatch
# from hmmlearn import hmm
from hmmlearn.hmm import CategoricalHMM
# 定义模型参数
pi = np.array([0.3, 0.7])  # 初始状态概率
A = np.array([[0.8, 0.2],
              [0.4, 0.6]])  # 状态转移概率矩阵
B = np.array([[0.1, 0.4, 0.5],
              [0.5, 0.4, 0.1]])  # 发射概率矩阵

# 定义观测序列
observation_symbols = ['S', 'G', 'L']
O = np.array(['S', 'G', 'L'])
observation_indices = np.array([observation_symbols.index(obs) for obs in O])  # 转换为索引

# 将观测序列转换为索引
observation_indices = np.array([0, 1, 2])  # S, G, L 对应的索引

# Directly calculate the most likely state transition
def most_likely_transition(pi, A, B, O, observation_indices):
    N = len(pi)  # Number of states
    T = len(O)   # Length of observation sequence

    # Initialize variables to track the most likely transition
    max_prob = 0
    best_transition = None

    # Iterate over all possible state transitions
    for t in range(T - 1):
        for src in range(N):
            for dest in range(N):
                # Calculate the probability of the transition
                prob = pi[src] * B[src, observation_indices[t]] * A[src, dest] * B[dest, observation_indices[t + 1]]
                if prob > max_prob:
                    max_prob = prob
                    best_transition = (src, dest)

    return best_transition, max_prob

# Calculate the most likely transition
best_transition, max_prob = most_likely_transition(pi, A, B, O, observation_indices)
print(f"Most likely transition: {best_transition} with probability {max_prob:.4f}")

# 使用 hmmlearn 定义 HMM 模型
# 使用 CategoricalHMM 定义 HMM 模型
model = CategoricalHMM(n_components=2, init_params="")  # 2 个隐藏状态
model.startprob_ = pi  # 设置初始状态概率
model.transmat_ = A  # 设置状态转移概率矩阵
model.emissionprob_ = B  # 设置发射概率矩阵

# 使用 Viterbi 算法计算最可能的状态序列
logprob, state_sequence = model.decode(observation_indices.reshape(-1, 1), algorithm="viterbi")
print("HMM learn Viterbi algorithm:")
print(f"Most likely state sequence (Viterbi): {state_sequence} with  probability {np.exp(logprob):.4f}")



def draw_viterbi_trellis(pi, A, B, O, observation_indices, best_path, logprob):
    N = len(pi)
    T = len(O)

    fig, ax = plt.subplots(figsize=(12, 6))

    # State positions
    state_y = {0: 1.5, 1: 0}  # Vertical positions for states
    node_radius = 0.1

    # Draw states and observations
    for t in range(T):
        for state in range(N):
            ax.text(t, state_y[state], f"S{state+1}", ha='center', va='center', fontsize=12)
        ax.text(t, -0.5, f"Obs {t+1}:\n{O[t]}", ha='center', va='center', fontsize=10, color='red')

    # Draw transitions
    for t in range(T - 1):
        for src in range(N):
            for dest in range(N):
                if A[src, dest] > 0:
                    start_x, start_y = t, state_y[src]
                    end_x, end_y = t + 1, state_y[dest]
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        arrowstyle='->', mutation_scale=15, color='black', lw=1.5
                    )
                    ax.add_patch(arrow)

    # Highlight the best path
    for t in range(T - 1):
        src = best_path[t]
        dest = best_path[t + 1]
        start_x, start_y = t, state_y[src]
        end_x, end_y = t + 1, state_y[dest]
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='->', mutation_scale=15, color='blue', lw=2.5
        )
        ax.add_patch(arrow)

         # Add probability to the best path
        prob = np.exp(logprob)  # Convert log probability to normal probability
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2 + 0.05
        ax.text(mid_x, mid_y, f"{prob:.4f}", ha='center', va='center', fontsize=10, color='blue')

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-1, 2)
    ax.set_title("Viterbi Trellis Diagram with Probabilities", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("pics/viterbi_trellis.png", dpi=300)
    plt.close()
    # plt.show()

# Draw the Viterbi trellis
draw_viterbi_trellis(pi, A, B, O, observation_indices, state_sequence, logprob)




# 计算观测序列的总概率
log_likelihood = model.score(observation_indices.reshape(-1, 1))
print(f"Likelihood of the observation sequence: {np.exp(log_likelihood):.4f}")