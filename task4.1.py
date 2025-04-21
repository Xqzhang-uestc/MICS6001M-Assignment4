import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)

# ========== 4.1.1 相同高斯分布 ==========
mu_same = 15
sigma_same = 2

# ========== 4.1.2 不同高斯分布 ==========
mu1_diff, sigma1_diff = 10.2, 0.5
mu2_diff, sigma2_diff = 8.5, 1.5

# 定义高斯PDF相乘函数
def multiply_gaussians(mu1, sigma1, mu2, sigma2):
    """计算两个高斯PDF相乘的结果"""
    mu = (mu1 * sigma2**2 + mu2 * sigma1**2) / (sigma1**2 + sigma2**2)
    sigma = np.sqrt((sigma1**2 * sigma2**2) / (sigma1**2 + sigma2**2))
    return mu, sigma

# ========== 绘图设置 ==========
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# ========== 绘图：4.1.1 相同高斯 ==========
plt.figure(figsize=(16, 12))
x_vals = np.linspace(0, 30, 1000)

# 原始PDF
pdf_X1_same = norm.pdf(x_vals, mu_same, sigma_same)
pdf_X2_same = norm.pdf(x_vals, mu_same, sigma_same)

# 加法部分
plt.subplot(2, 2, 1)
plt.plot(x_vals, pdf_X1_same, label="X1 ~ N(15, 2)")
plt.plot(x_vals, pdf_X2_same, label="X2 ~ N(15, 2)")
plt.plot(x_vals, norm.pdf(x_vals, mu_same*2, np.sqrt(2)*sigma_same),
         'k--', label="X1+X2 ~ N(30, 2.83)")
plt.title('Addition of Same Gaussians')
plt.legend()
plt.grid(True)

# PDF相乘部分
mu_mul_same, sigma_mul_same = multiply_gaussians(mu_same, sigma_same, mu_same, sigma_same)
pdf_mul_same = norm.pdf(x_vals, mu_mul_same, sigma_mul_same)

plt.subplot(2, 2, 2)
plt.plot(x_vals, pdf_X1_same, label="X1 ~ N(15, 2)")
plt.plot(x_vals, pdf_X2_same, label="X2 ~ N(15, 2)")
plt.plot(x_vals, pdf_mul_same, 'k--',
         label=f"X1*X2 ~ N({mu_mul_same:.1f}, {sigma_mul_same:.2f})")
plt.title('Multiplication of Same Gaussian PDFs')
plt.legend()
plt.grid(True)

# ========== 绘图：4.1.2 不同高斯 ==========
x_vals_diff = np.linspace(0, 25, 1000)

# 原始PDF
pdf_X1_diff = norm.pdf(x_vals_diff, mu1_diff, sigma1_diff)
pdf_X2_diff = norm.pdf(x_vals_diff, mu2_diff, sigma2_diff)

# 加法部分
plt.subplot(2, 2, 3)
plt.plot(x_vals_diff, pdf_X1_diff, label="X1 ~ N(10.2, 0.5)")
plt.plot(x_vals_diff, pdf_X2_diff, label="X2 ~ N(8.5, 1.5)")
plt.plot(x_vals_diff, norm.pdf(x_vals_diff, mu1_diff+mu2_diff,
                              np.sqrt(sigma1_diff**2 + sigma2_diff**2)),
         'k--', label="X1+X2 ~ N(18.7, 1.58)")
plt.title("Addition of Different Gaussians")
plt.legend()
plt.grid(True)

# PDF相乘部分
mu_mul_diff, sigma_mul_diff = multiply_gaussians(mu1_diff, sigma1_diff, mu2_diff, sigma2_diff)
pdf_mul_diff = norm.pdf(x_vals_diff, mu_mul_diff, sigma_mul_diff)

plt.subplot(2, 2, 4)
plt.plot(x_vals_diff, pdf_X1_diff, label="X1 ~ N(10.2, 0.5)")
plt.plot(x_vals_diff, pdf_X2_diff, label="X2 ~ N(8.5, 1.5)")
plt.plot(x_vals_diff, pdf_mul_diff, 'k--',
         label=f"X1*X2 ~ N({mu_mul_diff:.2f}, {sigma_mul_diff:.2f})")
plt.title("Multiplication of Different Gaussian PDFs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('pics/task4_gaussian_pdf_operations.png', dpi=350)
plt.show()