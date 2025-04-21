import numpy as np
from hmmlearn import hmm

# Define HMM parameters
n_states = 2  # Fair (0) and Loaded (1)
n_observations = 6  # Die faces 1-6

# Initial state probabilities (pi)
start_prob = np.array([0.99, 0.01])  # [P(fair), P(loaded)]

# Transition matrix (A)
trans_mat = np.array([
    [0.99, 0.01],  # Fair -> [Fair, Loaded]
    [0.99, 0.01]   # Loaded -> [Fair, Loaded]
])

# Emission probabilities (B)
emission_probs = np.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],  # Fair die
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]   # Loaded die
])

# Create HMM model
model = hmm.CategoricalHMM(n_components=n_states, init_params="")
model.startprob_ = start_prob
model.transmat_ = trans_mat
model.emissionprob_ = emission_probs

# Example usage:
# 1. Generate sample sequences
X, Z = model.sample(n_samples=100)  # Returns (observations, hidden_states)
print("First 10 observations:", X[:10].flatten())
print("First 10 hidden states:", Z[:10])

# 2. Predict hidden states for a sequence
test_seq = np.array([[4], [5], [5], [5], [0]])  # Observations must be 2D
logprob, hidden_states = model.decode(test_seq)
print("\nTest sequence:", test_seq.flatten() + 1)
print("Predicted hidden states:", hidden_states)
print("Log probability:", logprob)

# 3. Compute likelihood of a sequence
print("\nP([6,6,6]) =", np.exp(model.score(np.array([[5],[5],[5]]))))

# Task 3.4.5
print("\nTask 3.4.5:")
# Given observation sequence
observations = np.array([[0], [0], [0], [0], 
                         [5], [5], [5], [5],
                         [0], [0], [0], [0],
                         [5], [5], [5], [5]])

# (1) Calculate sequence likelihood
log_likelihood = model.score(observations)
likelihood = np.exp(log_likelihood)
print(f"(1) Sequence likelihood: {likelihood:.8f} (logprob: {log_likelihood:.4f})")

# (2) Find most likely state sequence
state_sequence = model.predict(observations)
state_names = ["Fair", "Loaded"]
print("\n(2) Most likely state sequence:")
print("Observations:", observations.flatten() + 1)
print("States:     ", [state_names[s] for s in state_sequence])

# Calculate probability of this state sequence
# (Need to compute path probability manually)
path_prob = model.startprob_[state_sequence[0]]  # Initial state
for i in range(1, len(state_sequence)):
    path_prob *= model.transmat_[state_sequence[i-1], state_sequence[i]]
for i in range(len(observations)):
    path_prob *= model.emissionprob_[state_sequence[i], observations[i]-1]

# print(type(path_prob))  
path_prob = path_prob.item()  # Convert to float for better readability  
print(f"\nProbability of this state sequence: {path_prob:.12f}")

# 使用对数概率计算路径概率
log_path_prob = np.log(model.startprob_[state_sequence[0]])  # 初始状态概率的对数
for i in range(1, len(state_sequence)):
    log_path_prob += np.log(model.transmat_[state_sequence[i-1], state_sequence[i]])  # 转移概率的对数
for i in range(len(observations)):
    log_path_prob += np.log(model.emissionprob_[state_sequence[i], observations[i]-1])  # 发射概率的对数

log_path_prob = log_path_prob.item()  # Convert to float for better readability
print(f"\nLog probability of this state sequence: {log_path_prob:.12f}")
# # 将对数概率转换回普通概率
# path_prob = np.exp(log_path_prob)
# path_prob = path_prob.item()  # Convert to float for better readability 
# print(f"\nProbability of this state sequence: {path_prob:.12f}")