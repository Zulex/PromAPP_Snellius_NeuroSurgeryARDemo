import numpy as np

class KalmanFilter3D:
    def __init__(self, initial_state, process_noise=1e-5, measurement_noise=1e-2, estimation_error=1.0):
        # Initial state [x, y, z]
        self.state = np.array(initial_state, dtype=float)

        # State covariance matrix (initially set to a large value)
        self.P = np.eye(3) * estimation_error

        # Process noise covariance matrix
        self.Q = np.eye(3) * process_noise

        # Measurement noise covariance matrix
        self.R = np.eye(3) * measurement_noise

        # State transition matrix (Identity since we assume constant position)
        self.F = np.eye(3)

        # Measurement matrix (Identity since we measure the position directly)
        self.H = np.eye(3)

    def predict(self):
        # Predict the next state (no change since the point is assumed to be static)
        self.state = np.dot(self.F, self.state)

        # Predict the error covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.state

    def update(self, measurement):
        # Measurement residual
        y = measurement - np.dot(self.H, self.state)

        # Residual covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state with the new measurement
        self.state = self.state + np.dot(K, y)

        # Update the error covariance
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        return self.state
    
def dynamic_adjustment(kalman_filter, previous_position, current_measurement, useDynamicAdjustment = True):
    
    if(useDynamicAdjustment):
        threshold = 0.03  # Threshold to detect significant movement
        movement = np.linalg.norm(current_measurement - previous_position)

        if movement > threshold:
            # Object is moving, make the filter more responsive
            kalman_filter.Q = 1e-1 * np.eye(3)
            kalman_filter.R = 1e-5 * np.eye(3)
            kalman_filter.P = 1e-1 * np.eye(3)  

        else:
            # Object is static, focus on noise suppression
            kalman_filter.Q = 1e-1 * np.eye(3)
            kalman_filter.R = 1e-6 * np.eye(3)



    return kalman_filter.update(current_measurement)

