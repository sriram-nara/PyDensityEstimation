import numpy as np
from filters.KalmanFilter import KalmanFilter
import matplotlib.pyplot as plt

# Simulation parameters
true_state = np.array([10, -5])  # True initial state [position, velocity]
n_steps = 50
dt = 1.0

# Covariances
measurement_noise_cov = np.array([[0.25, 0], [0, 0.25]])
process_noise_cov = np.array([[0.1, 0], [0, 0.1]])

# State transition and observation matrices
F = np.array([[1, dt],
                [0, 1]])
H = np.eye(2)

# Initial guess
initial_state_estimate = np.zeros(2)
initial_covariance_estimate = np.eye(2) * 10

# Generate measurements and true states
true_states = np.zeros((n_steps, 2))
measurements = np.zeros((n_steps, 2))
for step in range(n_steps):
    # Simulate process (constant acceleration: gravity)
    true_state = F @ true_state + np.array([0, -9.81 * dt])
    true_states[step] = true_state
    measurements[step] = true_state + np.random.multivariate_normal(np.zeros(2), measurement_noise_cov)

# Run Kalman filter
kf = KalmanFilter(
    x0=initial_state_estimate,
    P0=initial_covariance_estimate,
    Q=process_noise_cov,
    R=measurement_noise_cov
)
x_est, P_est = kf.kalman_filter(measurements, F, H)

# Extended Kalman Filter setup (linear system, so f and h are linear)
def f(x):
    return F @ x

def h(x):
    return H @ x

def F_jacobian(x):
    return F

def H_jacobian(x):
    return H

x_est_ekf, P_est_ekf = kf.extended_kalman_filter(measurements, f, h, F_jacobian, H_jacobian)

# Unscented Kalman Filter setup (linear system, so f and h are linear)
x_est_ukf, P_est_ukf = kf.unscented_kalman_filter(measurements, f, h)

# Calculate errors and covariances for EKF
position_errors_ekf = np.abs(true_states[:, 0] - x_est_ekf[0, :])
velocity_errors_ekf = np.abs(true_states[:, 1] - x_est_ekf[1, :])
covariances_ekf = np.array([np.diag(P_est_ekf[:, :, i]) for i in range(n_steps)])

# Calculate errors and covariances for UKF
position_errors_ukf = np.abs(true_states[:, 0] - x_est_ukf[0, :])
velocity_errors_ukf = np.abs(true_states[:, 1] - x_est_ukf[1, :])
covariances_ukf = np.array([np.diag(P_est_ukf[:, :, i]) for i in range(n_steps)])

# Calculate errors and covariances for KF
position_errors = np.abs(true_states[:, 0] - x_est[0, :])
velocity_errors = np.abs(true_states[:, 1] - x_est[1, :])
covariances = np.array([np.diag(P_est[:, :, i]) for i in range(n_steps)])

# Plotting
fig, axs = plt.subplots(4, 3, figsize=(20, 20))

# KF plots
axs[0, 0].plot(true_states[:, 0], label='True Positions')
axs[0, 0].plot(measurements[:, 0], label='Measured Positions', linestyle='--')
axs[0, 0].set_title('KF: Position over Time')
axs[0, 0].legend()

axs[1, 0].plot(x_est[0, :], label='Estimated Positions', color='orange')
axs[1, 0].set_title('KF: Estimated Position Over Time')

axs[2, 0].plot(position_errors, label='Position Errors', color='red')
axs[2, 0].set_title('KF: Position Error Over Time')

axs[3, 0].plot(covariances[:, 0], label='Covariance of Positions', color='purple')
axs[3, 0].set_title('KF: Covariance of Position Estimate Over Time')

# EKF plots
axs[0, 1].plot(true_states[:, 0], label='True Positions')
axs[0, 1].plot(measurements[:, 0], label='Measured Positions', linestyle='--')
axs[0, 1].set_title('EKF: Position over Time')
axs[0, 1].legend()

axs[1, 1].plot(x_est_ekf[0, :], label='Estimated Positions', color='orange')
axs[1, 1].set_title('EKF: Estimated Position Over Time')

axs[2, 1].plot(position_errors_ekf, label='Position Errors', color='red')
axs[2, 1].set_title('EKF: Position Error Over Time')

axs[3, 1].plot(covariances_ekf[:, 0], label='Covariance of Positions', color='purple')
axs[3, 1].set_title('EKF: Covariance of Position Estimate Over Time')

# UKF plots
axs[0, 2].plot(true_states[:, 0], label='True Positions')
axs[0, 2].plot(measurements[:, 0], label='Measured Positions', linestyle='--')
axs[0, 2].set_title('UKF: Position over Time')
axs[0, 2].legend()

axs[1, 2].plot(x_est_ukf[0, :], label='Estimated Positions', color='orange')
axs[1, 2].set_title('UKF: Estimated Position Over Time')

axs[2, 2].plot(position_errors_ukf, label='Position Errors', color='red')
axs[2, 2].set_title('UKF: Position Error Over Time')

axs[3, 2].plot(covariances_ukf[:, 0], label='Covariance of Positions', color='purple')
axs[3, 2].set_title('UKF: Covariance of Position Estimate Over Time')

plt.tight_layout()
plt.show()

# Print the results of the test case
print("True state over time:\n", true_states[:, 0])
print("\nMeasured positions over time:\n", measurements[:, 0])
print("\nKF Estimated positions over time:\n", x_est[0, :])
print("\nEKF Estimated positions over time:\n", x_est_ekf[0, :])
print("\nUKF Estimated positions over time:\n", x_est_ukf[0, :])
