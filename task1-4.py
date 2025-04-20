import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Prepare data (convert to 2D array of shape (n_samples, n_features))
observations = np.array([
    [50, 50], [60, 60], [70, 70], [80, 70], [90, 60],
    [100, 50], [100, 40], [50, 50], [40, 40], [50, 70],
    [50, 50], [60, 60], [70, 70], [80, 70], [90, 60],
    [100, 50], [100, 40], [50, 50], [40, 40], [50, 70]
])

# 1. Two-state HMM (Normal/Abnormal)
model_2state = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000,  init_params="", params="stmc")

# Initialize with problem parameters
model_2state.startprob_ = np.array([1.0, 0.0])  # Starts in normal state
model_2state.means_ = np.array([[50, 50],  # Normal state means
                               [90, 50]])  # Abnormal state means
model_2state.fit(observations)  # Learn transition matrix and covariances

# Results for two-state model
print("(1-3) Two-State Model Results:")
print(f"1. Sequence log likelihood: {model_2state.score(observations):.4e}")
print("2. Most likely state sequence (0=Normal, 1=Abnormal):")
print(model_2state.predict(observations))
print("3. Transition matrix:")
print(model_2state.transmat_)
print("Covariance matrices:")
print(model_2state.covars_)

# 2. Three-state HMM (adding Uncertain state)
model_3state = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, init_params="", params="stmc")
model_3state.startprob_ = np.array([1.0, 0.0, 0.0])
model_3state.means_ = np.array([[50, 50],  # Normal
                                [90, 50],  # Abnormal
                                [70, 60]]) # Uncertain
model_3state.fit(observations)

print("\n(4) Three-State Model Results:")
print(f"1. Sequence log likelihood: {model_3state.score(observations):.4e}")
print("2. Most likely state sequence (0=Normal, 1=Abnormal, 2=Uncertain):")
print(model_3state.predict(observations))
print("Transition matrix:")
print(model_3state.transmat_)
print("Means:")
print(model_3state.means_)
print("Covariance matrices:")
print(model_3state.covars_)

# Visualization
# plt.figure(figsize=(12, 6))
# plt.plot(observations[:,0], 'r-', label='Xc (Center)')
# plt.plot(observations[:,1], 'b-', label='Xb (Boundary)')
# plt.plot(model_2state.predict(observations), 'g--', label='State (2-state)')
# plt.plot(model_3state.predict(observations), 'k:', label='State (3-state)')
# plt.legend()
# plt.xlabel("Time step")
# plt.ylabel("Temperature / State")
# plt.title("Temperature Observations with Predicted States")
# plt.show()

# Calculate the log likelihood of the observations for both models
log_likelihood_2state = model_2state.score(observations)
log_likelihood_3state = model_3state.score(observations)
print(f"Log Likelihood (2-state): {log_likelihood_2state:.4f}")
print(f"Log Likelihood (3-state): {log_likelihood_3state:.4f}")

# Calculate AIC and BIC for both models
n_params_2state = model_2state.n_components * (model_2state.n_features + 1) + model_2state.n_components**2
n_params_3state = model_3state.n_components * (model_3state.n_features + 1) + model_3state.n_components**2

aic_2state = 2 * n_params_2state - 2 * log_likelihood_2state
aic_3state = 2 * n_params_3state - 2 * log_likelihood_3state

bic_2state = n_params_2state * np.log(len(observations)) - 2 * log_likelihood_2state
bic_3state = n_params_3state * np.log(len(observations)) - 2 * log_likelihood_3state

print(f"AIC (2-state): {aic_2state:.4f}, BIC (2-state): {bic_2state:.4f}")
print(f"AIC (3-state): {aic_3state:.4f}, BIC (3-state): {bic_3state:.4f}")

# Visualize the teperature data
plt.figure(figsize=(12, 6))
plt.plot(observations[:, 0], 'r-', label='Xc (Center)')
plt.plot(observations[:, 1], 'b-', label='Xb (Boundary)')
plt.title("Temperature Observations")
plt.xlabel("Time step")
plt.ylabel("Temperature")
plt.legend()
# plt.show()

# Visualize the predicted states
plt.figure(figsize=(12, 6))
plt.plot(model_2state.predict(observations), 'g--', label='State (2-state)')
plt.plot(model_3state.predict(observations), 'k:', label='State (3-state)')
plt.title("Predicted States")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()
# plt.show()

# Visualize the predicted states with observations
plt.figure(figsize=(12, 6))
plt.plot(observations[:, 0], 'r-', label='Xc (Center)')
plt.plot(observations[:, 1], 'b-', label='Xb (Boundary)')
plt.plot(model_2state.predict(observations), 'g--', label='State (2-state)')
plt.plot(model_3state.predict(observations), 'k:', label='State (3-state)')
plt.title("Predicted States vs Observations")
plt.xlabel("Time step")
plt.ylabel("Temperature / State")
plt.legend()
plt.show()

# # 获取预测的隐藏状态序列
# predicted_states_2state = model_2state.predict(observations)
# predicted_states_3state = model_3state.predict(observations)

# # 绘制散点图（2-state HMM）
# plt.figure(figsize=(12, 6))
# for state in range(2):  # 两个隐藏状态
#     state_indices = np.where(predicted_states_2state == state)  # 获取属于该状态的样本索引
#     plt.scatter(observations[state_indices, 0], observations[state_indices, 1], label=f"State {state} (2-state)")
# plt.title("Scatter Plot of Features (2-state HMM)")
# plt.xlabel("Xc (Center)")
# plt.ylabel("Xb (Boundary)")
# plt.legend()
# plt.grid()
# plt.show()

# # 绘制散点图（3-state HMM）
# plt.figure(figsize=(12, 6))
# for state in range(3):  # 三个隐藏状态
#     state_indices = np.where(predicted_states_3state == state)  # 获取属于该状态的样本索引
#     plt.scatter(observations[state_indices, 0], observations[state_indices, 1], label=f"State {state} (3-state)")
# plt.title("Scatter Plot of Features (3-state HMM)")
# plt.xlabel("Xc (Center)")
# plt.ylabel("Xb (Boundary)")
# plt.legend()
# plt.grid()
# plt.show()