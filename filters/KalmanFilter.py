import numpy as np

class KalmanFilter:
    def __init__(self, x0, P0, Q, R):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def kalman_filter(self, measurements, F, H):
        n = len(measurements)
        x_est = np.zeros((self.x.shape[0], n))
        P_est = np.zeros((self.P.shape[0], self.P.shape[1], n))
        x = self.x
        P = self.P

        for i in range(n):
            # Prediction
            x = F @ x
            P = F @ P @ F.T + self.Q

            # Update
            y = measurements[i] - H @ x
            S = H @ P @ H.T + self.R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(P.shape[0]) - K @ H) @ P

            x_est[:, i] = x
            P_est[:, :, i] = P

        return x_est, P_est

    def extended_kalman_filter(self, measurements, f, h, F_jacobian, H_jacobian):
        n = len(measurements)
        x_est = np.zeros((self.x.shape[0], n))
        P_est = np.zeros((self.P.shape[0], self.P.shape[1], n))
        x = self.x
        P = self.P

        for i in range(n):
            # Prediction
            x = f(x)
            F = F_jacobian(x)
            P = F @ P @ F.T + self.Q

            # Update
            H = H_jacobian(x)
            y = measurements[i] - h(x)
            S = H @ P @ H.T + self.R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(P.shape[0]) - K @ H) @ P

            x_est[:, i] = x
            P_est[:, :, i] = P

        return x_est, P_est

    def unscented_kalman_filter(self, measurements, f, h, alpha=1, beta=2, kappa=None):
        n = len(measurements)
        L = self.x.shape[0]
        if kappa is None:
            kappa = 3 - L
        lam = alpha ** 2 * (L + kappa) - L

        Wm, Wc = self._unscented_weights(L, alpha, beta, kappa)
        x = self.x
        P = self.P
        x_est = np.zeros((L, n))
        P_est = np.zeros((L, L, n))

        for i in range(n):
            # Sigma points
            X = self._sigma_points(x, P, lam)
            # Prediction
            X_pred = np.array([f(Xj) for Xj in X])
            x_pred = np.sum(Wm[:, None] * X_pred, axis=0)
            P_pred = self.Q.copy()
            for j in range(2 * L + 1):
                y = X_pred[j] - x_pred
                P_pred += Wc[j] * np.outer(y, y)

            # Measurement prediction
            Y = np.array([h(Xj) for Xj in X_pred])
            y_pred = np.sum(Wm[:, None] * Y, axis=0)
            Pyy = self.R.copy()
            for j in range(2 * L + 1):
                y = Y[j] - y_pred
                Pyy += Wc[j] * np.outer(y, y)
            Pxy = np.zeros((L, Y.shape[1]))
            for j in range(2 * L + 1):
                Pxy += Wc[j] * np.outer(X_pred[j] - x_pred, Y[j] - y_pred)

            # Update
            K = Pxy @ np.linalg.inv(Pyy)
            x = x_pred + K @ (measurements[i] - y_pred)
            P = P_pred - K @ Pyy @ K.T

            x_est[:, i] = x
            P_est[:, :, i] = P

        return x_est, P_est

    def _unscented_weights(self, L, alpha, beta, kappa):
        lam = alpha ** 2 * (L + kappa) - L
        Wm = np.full(2 * L + 1, 1 / (2 * (L + lam)))
        Wc = np.full(2 * L + 1, 1 / (2 * (L + lam)))
        Wm[0] = lam / (L + lam)
        Wc[0] = lam / (L + lam) + (1 - alpha ** 2 + beta)
        return Wm, Wc

    def _sigma_points(self, x, P, lam):
        L = x.shape[0]
        S = np.linalg.cholesky((L + lam) * P)
        X = np.zeros((2 * L + 1, L))
        X[0] = x
        for i in range(L):
            X[i + 1] = x + S[:, i]
            X[L + i + 1] = x - S[:, i]
        return X
    


