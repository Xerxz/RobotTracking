from pykalman import KalmanFilter
import numpy as np

global kalman_filters


def init_kalman_filter(video_fps=30):
    global kalman_filters
    n_dim_state = 2
    n_dim_obs = 1

    # Initialize a list of Kalman filters for each vehicle
    kalman_filters = [
        KalmanFilter(initial_state_mean=[0, 0], n_dim_obs=1),
        KalmanFilter(initial_state_mean=[0, 0], n_dim_obs=1),
        KalmanFilter(initial_state_mean=[0, 0], n_dim_obs=1)
    ]

    # Set the initial state mean and covariance for each Kalman filter
    initial_state_covariance = np.eye(2) * 1e4  # Use a high initial value to represent uncertainty
    process_noise_covariance = np.eye(2) * 0.01  # Adjust this value to control the smoothness of the angle estimation
    measurement_noise_covariance = 10  # Adjust this value based on the noise level in the angle measurements
    dt = 1 / video_fps  # time between frames
    measurement_matrix = [[1, 0]]  # Only measure angle
    measurement_covariance = np.array([[measurement_noise_covariance]])
    transition_matrix = [[1, dt],
                         [0, 1]]

    for kf in kalman_filters:
        kf.__init__(initial_state_mean=np.zeros(n_dim_state),
                    initial_state_covariance=initial_state_covariance,
                    transition_matrices=transition_matrix,
                    transition_covariance=process_noise_covariance,
                    observation_matrices=measurement_matrix,  # Only measure angle
                    observation_covariance=measurement_covariance,
                    transition_offsets=np.zeros(n_dim_state),
                    observation_offsets=np.zeros(n_dim_obs))
