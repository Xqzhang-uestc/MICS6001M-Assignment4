# MICS6001M Assignment 4

This repository contains the implementation of Gaussian distribution operations and Kalman filter experiments for **MICS6001M Assignment 4**.

## Repository Structure

### Code Files:
1. **`task4.1.py`**: 
   - Performs addition and multiplication of Gaussian distributions.
   - Visualizes results for same and different Gaussian distributions.
   - Output: `pics/task4_gaussian_pdf_operations.png`

2. **`task4.2.py`**:
   - Implements Kalman Filter in two ways:
     - Without explicit Kalman gain.
     - With explicit Kalman gain.
   - Compares the two implementations.
   - Output: `pics/task4_kf_implementation.png`

3. **`task4.3.py`**:
   - Conducts experiments with Kalman Filter under various conditions:
     - Large initial error.
     - High measurement noise.
     - Changing velocity.
   - Output: `pics/task4_kf_variance.png`

### Output Images:
- `pics/task4_gaussian_pdf_operations.png`: Results of Gaussian operations.
- `pics/task4_kf_implementation.png`: Comparison of Kalman Filter implementations.
- `pics/task4_kf_variance.png`: Results from different Kalman Filter experiments.

## Requirements

- Python `>=3.6`
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`

Install dependencies using:
```bash
pip install numpy matplotlib scipy