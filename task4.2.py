import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class KalmanFilter:
    def __init__(self, initial_pos, initial_uncertainty, process_noise, measurement_noise):
        self.x = initial_pos # 预测初始位置
        self.P = initial_uncertainty # 预测初始方差
        self.Q = process_noise # 预测过程噪声
        self.R = measurement_noise # 测量噪声（测量方差）
        self.F = 1  # 状态转移矩阵描述状态如何从上一步变到下一步，=1 表示匀速或无动力过程
        self.H = 1  # 观测矩阵描述状态如何映射成可以观测的量，这里直接观测位置，所以=1
        # 记录中间结果
        self.predicted_states = [] #存储每一步预测的状态（先验）
        self.updated_states = [] #存储每一步更新后的状态（后验）
        self.variances = [] #存储每一步更新后的方差（后验）

    def predict_without_K(self):
        self.x = self.F * self.x # 预测当前均值
        self.P = self.F * self.P * self.F + self.Q # 预测当前方差
        self.predicted_states.append(self.x) #记录预测值
        return self.x, self.P

    def update_without_K(self, z):
        y = z - self.H * self.x  # 观测残差：测量值-预测值
        S = self.H * self.P * self.H + self.R  # 预测的观测方差（S ≈ σ^2 + σ_z^2）
        K_temp = self.P * self.H / S  # σ^2 / (σ^2 + σ_z^2)
        self.x += K_temp * y    # 更新均值：结合 prior 和 z 的加权平均
        self.P *= (1 - K_temp * self.H)  #  更新方差：更新为合并后的新 σ^2
        self.updated_states.append(self.x)
        self.variances.append(self.P)
        return K_temp

    def predict_with_K(self):
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q
        self.predicted_states.append(self.x)
        return self.x, self.P

    def update_with_K(self, z):
        K = self.P * self.H / (self.H * self.P * self.H + self.R) # 卡尔曼增益
        self.x += K * (z - self.H * self.x) #estimate = prediction + K·(measurement −prediction)
        self.P *= (1 - K * self.H)
        self.updated_states.append(self.x)
        self.variances.append(self.P)
        return K


def plot_comparison(true, meas, pred1, est1, var1, pred2, est2, var2):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    plt.figure(figsize=(16, 6))
    t = range(len(true))

    # 左侧：不使用Kalman gain的显式实现
    plt.subplot(1, 2, 1)
    plt.plot(t, true, 'g-', linewidth=2, label='True Position')
    plt.plot(t, meas, 'ro', markersize=4, alpha=0.5, label='Measurements')
    plt.plot(t, pred1, 'b--', linewidth=1, label='Predicted')
    plt.plot(t, est1, 'k-', linewidth=2, label='KF Estimate')
    plt.fill_between(t,
                     np.array(est1) - 2 * np.sqrt(var1),
                     np.array(est1) + 2 * np.sqrt(var1),
                     color='yellow', alpha=0.2, label='2σ Variance')
    plt.title('KF Implementation Without Explicit Kalman Gain')
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    # 右侧：使用Kalman gain的实现
    plt.subplot(1, 2, 2)
    plt.plot(t, true, 'g-', linewidth=2, label='True Position')
    plt.plot(t, meas, 'ro', markersize=4, alpha=0.5, label='Measurements')
    plt.plot(t, pred2, 'b--', linewidth=1, label='Predicted')
    plt.plot(t, est2, 'k-', linewidth=2, label='KF Estimate')
    plt.fill_between(t,
                     np.array(est2) - 2 * np.sqrt(var2),
                     np.array(est2) + 2 * np.sqrt(var2),
                     color='yellow', alpha=0.2, label='2σ Variance')
    plt.title('KF Implementation With Kalman Gain')
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('pics/task4_kf_implementation.png', dpi=300)
    plt.show()


# 模拟参数
true_pos = 0
true_vel = 0.5
num_steps = 50
dt = 1

# 生成数据
true_positions = np.cumsum([true_vel * dt] * num_steps)
measurements = true_positions + np.random.normal(0, 1, num_steps)

# 初始化滤波器
kf_without_K = KalmanFilter(initial_pos=0, initial_uncertainty=1,
                            process_noise=0.1, measurement_noise=1)
kf_with_K = KalmanFilter(initial_pos=0, initial_uncertainty=1,
                         process_noise=0.1, measurement_noise=1)

# 运行滤波
for z in measurements:
    kf_without_K.predict_without_K()
    kf_without_K.update_without_K(z)

    kf_with_K.predict_with_K()
    kf_with_K.update_with_K(z)

# 绘图比较
plot_comparison(true_positions, measurements,
                kf_without_K.predicted_states, kf_without_K.updated_states, kf_without_K.variances,
                kf_with_K.predicted_states, kf_with_K.updated_states, kf_with_K.variances)