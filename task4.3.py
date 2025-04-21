import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, initial_pos, initial_uncertainty, process_noise, measurement_noise):
        self.x = initial_pos
        self.P = initial_uncertainty
        self.Q = process_noise
        self.R = measurement_noise
        self.velocity = 10  # 初始速度 (m/s)
        self.dt = 1  # 时间步长 (s)
        self.predicted_states = []
        self.updated_states = []
        self.variances = []

    def predict(self):
        self.x += self.velocity * self.dt
        self.P += self.Q
        self.predicted_states.append(self.x)
        return self.x, self.P

    def update(self, z):
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        self.updated_states.append(self.x)
        self.variances.append(self.P)
        return K


def run_experiments():
    # 设置随机种子保证结果可重复
    np.random.seed(42)

    # 创建3行1列的子图
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    fig.tight_layout(pad=4.0)

    # 实验1: 大初始误差
    true_pos1 = np.cumsum([10] * 50)
    measurements1 = true_pos1 + np.random.normal(0, np.sqrt(30), 50) #50个服从高斯分布的噪声项，均值为0，标准差为√30
    kf1 = KalmanFilter(initial_pos=1000, initial_uncertainty=1000,
                       process_noise=1, measurement_noise=30)
    for z in measurements1:
        kf1.predict()
        kf1.update(z)

    # 实验2: 大测量噪声
    true_pos2 = np.cumsum([10] * 50)
    measurements2 = true_pos2 + np.random.normal(0, np.sqrt(1500), 50) #100个服从高斯分布的噪声项，均值为0，标准差为√300
    kf2 = KalmanFilter(initial_pos=0, initial_uncertainty=1,
                       process_noise=2, measurement_noise=1500)
    for z in measurements2:
        kf2.predict()
        kf2.update(z)

    # 实验3: 速度变化
    true_pos3 = np.cumsum([10] * 50)
    velocities = [10 * (1.05) ** t for t in range(50)] # 速度每步增加5%
    measure_pos3 = np.cumsum(velocities)
    measurements3 = measure_pos3 + np.random.normal(0, 1, 50)
    kf3 = KalmanFilter(initial_pos=0, initial_uncertainty=1,
                       process_noise=0.1, measurement_noise=1)
    for i, z in enumerate(measurements3):
        #kf3.velocity = velocities[i]
        kf3.predict()
        kf3.update(z)

    # 绘制实验1结果
    t1 = range(len(true_pos1))
    #axs[0].plot(t1, true_pos1, 'g-', linewidth=2, label='True Position')
    axs[0].plot(t1, measurements1, 'r.', markersize=4, alpha=0.5, label='Measurements')
    axs[0].plot(t1, kf1.predicted_states, 'b--', linewidth=1, label='Predicted')
    axs[0].plot(t1, kf1.updated_states, 'k-', linewidth=2, label='KF Estimate')
    axs[0].fill_between(t1,
                        np.array(kf1.updated_states) - 2 * np.sqrt(kf1.variances),
                        np.array(kf1.updated_states) + 2 * np.sqrt(kf1.variances),
                        color='yellow', alpha=0.2, label='2σ Variance')
    axs[0].set_title('KF with Large Initial Error (1000 m)', fontsize=12)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Position (m)')
    axs[0].legend()
    axs[0].grid(True)

    # 绘制实验2结果
    t2 = range(len(true_pos2))
    #axs[1].plot(t2, true_pos2, 'g-', linewidth=2, label='True Position')
    axs[1].plot(t2, measurements2, 'r.', markersize=4, alpha=0.5, label='Measurements')
    axs[1].plot(t2, kf2.predicted_states, 'b--', linewidth=1, label='Predicted')
    axs[1].plot(t2, kf2.updated_states, 'k-', linewidth=2, label='KF Estimate')
    axs[1].fill_between(t2,
                        np.array(kf2.updated_states) - 2 * np.sqrt(kf2.variances),
                        np.array(kf2.updated_states) + 2 * np.sqrt(kf2.variances),
                        color='yellow', alpha=0.2, label='2σ Variance')
    axs[1].set_title('KF with High Measurement Noise (R=300)', fontsize=12)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Position (m)')
    axs[1].legend()
    axs[1].grid(True)

    # 绘制实验3结果
    t3 = range(len(true_pos3))
    #axs[2].plot(t3, true_pos3, 'g-', linewidth=2, label='True Position')
    axs[2].plot(t3, measurements3, 'r.', markersize=4, alpha=0.5, label='Measurements')
    axs[2].plot(t3, kf3.predicted_states, 'b--', linewidth=1, label='Predicted')
    axs[2].plot(t3, kf3.updated_states, 'k-', linewidth=2, label='KF Estimate')
    axs[2].fill_between(t3,
                        np.array(kf3.updated_states) - 2 * np.sqrt(kf3.variances),
                        np.array(kf3.updated_states) + 2 * np.sqrt(kf3.variances),
                        color='yellow', alpha=0.2, label='2σ Variance')
    axs[2].set_title('KF with Velocity Change (5%/step) and Low Process Noise (Q=0.1)', fontsize=12)
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Position (m)')
    axs[2].legend()
    axs[2].grid(True)

    # 保存图像
    plt.savefig('pics/task4_kf_variance.png', dpi=350)
    plt.show()


if __name__ == "__main__":
    run_experiments()