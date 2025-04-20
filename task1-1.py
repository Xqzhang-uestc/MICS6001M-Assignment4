import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.table import Table
from itertools import product
from matplotlib.patches import FancyArrowPatch
# 定义模型参数
pi = np.array([0.3, 0.7])  # 初始状态概率
A = np.array([[0.8, 0.2],
              [0.4, 0.6]])  # 状态转移概率矩阵
B = np.array([[0.1, 0.4, 0.5],
              [0.5, 0.4, 0.1]])  # 发射概率矩阵

# 定义观测序列
O = np.array(['S', 'G', 'L'])

# 将观测序列转换为索引
observation_indices = np.array([0, 1, 2])  # S, G, L 对应的索引



# A. 绘制状态转移图和发射图

def plot_hmm(pi, A, B, observations):
    """
    使用 matplotlib 绘制 HMM 状态转移图
    
    参数:
        pi: 初始状态概率向量
        A: 状态转移概率矩阵
        B: 发射概率矩阵
        observations: 观测符号列表
    """
    num_states = len(pi)
    num_observations = len(observations)
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加状态节点
    for i in range(num_states):
        G.add_node(f"State {i+1}")
    
    # 添加状态转移边
    for i in range(num_states):
        for j in range(num_states):
            if A[i, j] > 0:
                G.add_edge(f"State {i+1}", f"State {j+1}", weight=A[i, j])

    # 固定布局（State 1在左，State 2在右）
    pos = {'State 1': np.array([0, 0]), 'State 2': np.array([1, 0])}
    
    plt.figure(figsize=(10, 6))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='skyblue', edgecolors='black')
    
    # 绘制边（带弯曲效果）
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='->',
        arrowsize=25,
        width=2,
        connectionstyle='arc3,rad=0.25'  # 关键修改：曲线边
    )
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    
    for (u, v), label in edge_labels.items():
        # 计算边的中点坐标
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        
        # 根据边方向微调位置
        if u == 'State 1' and v == 'State 2':
            y += 0.08  # 上移标签
        elif u == 'State 2' and v == 'State 1':
            y -= 0.08  # 下移标签
        elif u == 'State 1' and v == 'State 1':
            y += 0.16
        elif u == 'State 2' and v == 'State 2':
            y += 0.16
        
        plt.text(
            x, y, label,
            fontsize=10,
            ha='center', va='center',
            
        )
    
    # 绘制初始概率
    for i in range(num_states):
        if pi[i] > 0:
            plt.text(pos[f"State {i+1}"][0], pos[f"State {i+1}"][1] - 0.03, 
                    f"Init: {pi[i]:.2f}", ha='center', va='center', fontsize=10)
    
    # 绘制状态标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    for i in range(num_states):
        table_x = pos[f"State {i+1}"][0]   # 调整表格位置
        table_y = pos[f"State {i+1}"][1] + 0.3   # 调整表格位置

        # 创建表格
        ax = plt.gca()
        table = Table(ax, bbox=[table_x - 0.1, table_y - 0.15, 0.3, 0.2])  # 调整表格大小和位置

        # 添加表头
        table.add_cell(0, 0, width=0.15, height=0.1, text="Obs", loc='center', facecolor='lightgray')
        table.add_cell(0, 1, width=0.15, height=0.1, text="Prob", loc='center', facecolor='lightgray')

        # 添加每个观测符号及其概率
        for k in range(num_observations):
            table.add_cell(k + 1, 0, width=0.15, height=0.1, text=observations[k], loc='center')
            table.add_cell(k + 1, 1, width=0.15, height=0.1, text=f"{B[i, k]:.2f}", loc='center')

        # 添加表格到图形
        ax.add_table(table)

    plt.title("HMM State Transition Diagram with Emissions", fontsize=14)
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.savefig("hmm_diagram.png", dpi=300)  # 保存图形为PNG文件
    plt.close()  # 关闭当前图形
    # plt.show()

# 绘制图形
# plot_hmm(pi, A, B, O)

# B. 直接计算观测序列的似然
def direct_likelihood(pi, A, B, observations, observation_symbols):
    """
    直接计算观测序列的似然（枚举所有状态路径）
    
    参数:
        pi: 初始状态概率向量 (shape: [N])
        A: 状态转移概率矩阵 (shape: [N, N])
        B: 发射概率矩阵 (shape: [N, M])
        observations: 观测序列 (如 ['S', 'G', 'L'])
        observation_symbols: 观测符号列表 (如 ['S', 'G', 'L'])
    
    返回:
        likelihood: 观测序列的似然概率
    """
    N = len(pi)  # 状态数
    T = len(observations)  # 观测序列长度
    
    # 将观测序列转换为索引
    obs_idx = [observation_symbols.index(obs) for obs in observations]
    
    # 生成所有可能的状态路径（长度为 T 的 N 元组）
    all_paths = product(range(N), repeat=T)
    
    likelihood = 0.0
    
    for path in all_paths:
        # 计算路径的联合概率 P(O, X | λ)
        prob = pi[path[0]] * B[path[0], obs_idx[0]]  # 初始状态和第一个观测
        
        for t in range(1, T):
            prob *= A[path[t-1], path[t]] * B[path[t], obs_idx[t]]  # 转移和发射
        
        likelihood += prob
    
    return likelihood

likelihood_directly = direct_likelihood(pi, A, B, O, ['S', 'G', 'L'])
print(f"Directly calculated likelihood: {likelihood_directly}")

# C. 绘制状态和观测转换的三元图

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

def plot_trellis_diagram(pi, A, B, observations, observation_symbols):
    N = len(pi)  # Number of states
    T = len(observations)  # Time steps
    obs_idx = [observation_symbols.index(obs) for obs in observations]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # State positions (S1 on top, S2 on bottom)
    state_y = {0: 1.5, 1: 0}  # Vertical positions
    node_radius = 0.1
    
    # Store node positions for precise arrow connections
    node_positions = {}
    for t in range(T):
        for state in range(N):
            node_positions[(t, state)] = (t, state_y[state])

    # Draw states (circles)
    for t in range(T):
        for state in range(N):
            ax.add_patch(Circle((t, state_y[state]), node_radius, 
                            color='skyblue', ec='black', zorder=3))
            ax.text(t, state_y[state], f"S{state+1}", 
                   ha='center', va='center', fontsize=12)

    # Draw observations
    for t in range(T):
        ax.text(t, -0.7, f"Obs {t+1}:\n{observations[t]}", 
                ha='center', va='center', fontsize=10, color='red')

    # Function to calculate arrow start/end points on node boundaries
    def get_arrow_points(src_t, src_state, dest_t, dest_state):
        src_x, src_y = node_positions[(src_t, src_state)]
        dest_x, dest_y = node_positions[(dest_t, dest_state)]
        
        # Calculate angle between nodes
        dx = dest_x - src_x
        dy = dest_y - src_y
        angle = np.arctan2(dy, dx)
        
        # Start point (on source node's circumference)
        start_x = src_x + node_radius * np.cos(angle)
        start_y = src_y + node_radius * np.sin(angle)
        
        # End point (on destination node's circumference)
        end_x = dest_x - node_radius * np.cos(angle)
        end_y = dest_y - node_radius * np.sin(angle)
        
        return (start_x, start_y), (end_x, end_y)

    # Draw transitions with precise boundary connections
    for t in range(T-1):
        for src in range(N):
            for dest in range(N):
                if A[src, dest] > 0:
                    (start_x, start_y), (end_x, end_y) = get_arrow_points(t, src, t+1, dest)
                    
                    # Determine curvature
                    curvature = 0.3 if src != dest else 0
                    
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        arrowstyle='->', 
                        mutation_scale=20,
                        color='black',
                        lw=1.5,
                        connectionstyle=f"arc3,rad={curvature}",
                        zorder=2
                    )
                    ax.add_patch(arrow)
                    
                    # Label positioning
                    x_mid = (start_x + end_x)/2
                    y_offset = 0.2 if src != dest else 0
                    y_dir = 1 if src < dest else -1
                    y_mid = (start_y + end_y)/2 + y_dir*y_offset
                    
                    ax.text(x_mid, y_mid, f"{A[src, dest]:.2f}", 
                            ha='center', va='center', 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                            zorder=4)

    for t in range(T):
        for state in range(N):
            prob = B[state, obs_idx[t]]
            ax.text(t, state_y[state] - 0.3, f"P({observations[t]}):{prob:.2f}", 
                    ha='center', va='top', fontsize=9, color='green')

    # Initial probabilities
    for state in range(N):
        ax.text(-0.2, state_y[state], f"Init:{pi[state]:.2f}", 
                ha='right', va='center', fontsize=10)

    ax.set_xlim(-0.8, T-0.2)
    ax.set_ylim(-1, 2)
    ax.set_title('HMM Trellis Diagram with Precise Arrow Connections', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("hmm_trellis_diagram.png", dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the current figure
    # plt.show()

observations = ['S', 'G', 'L']
observation_symbols = ['S', 'G', 'L']

# plot_trellis_diagram(pi, A, B, observations, observation_symbols)

# D. 使用前向算法计算观测序列的似然
def forward_algorithm(O, pi, A, B):
    N = len(pi)
    T = len(O)
    alpha = np.zeros((T+1, N))

    # 初始化
    for i in range(N):
        alpha[0, i] = pi[i] * B[i, observation_indices[0]]

    # 递归计算
    for t in range(1, T):
        for i in range(N):
            alpha[t, i] = np.sum(alpha[t-1] * A[:, i]) * B[i, observation_indices[t]]

    # 终止
    likelihood = np.sum(alpha[T-1])

    return likelihood

likelihood_forward = forward_algorithm(O, pi, A, B)
print(f"Likelihood using forward algorithm: {likelihood_forward}")

# E. 解释结果
print("\nExplanation of results:")
print(f"The directly calculated likelihood is: {likelihood_directly}")
print(f"The likelihood using forward algorithm is: {likelihood_forward}")
print("Both methods should give the same result, as they compute the same probability.")

# F. 比较两种计算方法的效率
def compare_efficiency(N, T):
    num_multiplications_direct = N**T * (N + 3*T - 1)
    num_multiplications_forward = (N**2 + N) * (T - 1) + N

    print(f"\nEfficiency comparison:")
    print(f"Number of multiplications for direct method: {num_multiplications_direct}")
    print(f"Number of multiplications for forward algorithm: {num_multiplications_forward}")
    print(f"Efficiency improvement of forward algorithm: {num_multiplications_direct / num_multiplications_forward:.2f} times")

compare_efficiency(2, 3)