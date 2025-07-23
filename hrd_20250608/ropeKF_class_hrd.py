import numpy as np
import orekit
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir
from os import path
import numpy as np
import pandas as pd
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.bodies import BodyShape, GeodeticPoint, OneAxisEllipsoid
from org.orekit.frames import Frame, FramesFactory, KinematicTransform
from org.orekit.models.earth.atmosphere import PythonAtmosphere
from org.orekit.time import AbsoluteDate, TimeScalesFactory, UTCScale
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import hrd_20250608.utilities_ds as u
import warnings
warnings.filterwarnings("ignore")


class SvdContainer:
    def __init__(self, u_svd, mu):
        self.U = u_svd
        self.mu = mu


class rope_filtering:
    '''
    Filtering of SINDY models based on TIEGCM physics based dataset
    '''

    def __init__(self, datapath: str = "hrd_20250608", drivers=None, selected_bf_dict=None, delta_rho_ic=0):
        # 72: longitude intervals, 36: latitude intervals, 45: altitude intervals
        self.input_data_sindy = np.load(path.join(
            datapath, 'z_drivers_dataset_hrd_02_09_std_rescaling_v13.npz'), allow_pickle=True)
        self.U0 = self.input_data_sindy['u_svd'].reshape(
            (72, 36, 45, 10), order='C')
        self.mu0 = self.input_data_sindy['mu_svd'].reshape(
            (72, 36, 45), order='C')
        self.models_coefficients = self.input_data_sindy['models_coefficients']
        self.initial_conditions = pd.DataFrame(self.input_data_sindy['initial_conditions'][()], columns=[
                                               'f10', 'kp']+[f'z_{str(k).zfill(2)}' for k in range(10)])
        if drivers is None:
            self.drivers = self.input_data_sindy['celestrack_drivers']
        else:
            self.drivers = drivers
            self.original_drivers = self.input_data_sindy['celestrack_drivers']

        self.X_reg_norm_dict_nl_dmd = self.input_data_sindy['models_coefficients'][(
        )]['X_reg_norm_dict_nl_dmd']
        self.Y_reg_norm_dict_nl_dmd = self.input_data_sindy['models_coefficients'][(
        )]['Y_reg_norm_dict_nl_dmd']

        # self.input_data_sindy['models_coefficients'][()]['q_low_f10_value']
        self.q_low_f10_value = 100.
        # self.input_data_sindy['models_coefficients'][()]['q_high_f10_value']
        self.q_high_f10_value = 160.

        self.x_train_svd_obj = SvdContainer(
            self.input_data_sindy['u_svd'], self.input_data_sindy['mu_svd'])
        self.x_train_svd_obj.norm_dict = {
            'x_mean': self.input_data_sindy['mu_svd']}
        self.normalization_method = 'std'
        T = [15.34175, 12.2734, 24., 48., 20.45566667, 30.6835, 61.367]
        self.delta_rho_ic = delta_rho_ic
        self.f10_idx = 5
        self.kp_idx = 6
        self.time_variables = 4
        self.pca_coupling = 4
        self.n_components = 10
        self.sub_intervals = 60
        self.sub_intervals_mins = 1
        self.kp_th = 0
        self.input_features = ['x_' + str(k+1).zfill(2) for k in range(10)]
        poly1 = {'p1': lambda x: x}
        poly1p5 = {'p1': lambda x: np.abs(x)**1.5}
        poly2 = {'p2': lambda x: x**2}
        poly3 = {'p3': lambda x: x**3}
        poly4 = {'p3': lambda x: x**4}
        poly5 = {'p5': lambda x: x**5}
        poly7 = {'p5': lambda x: x**7}
        exp1 = {'e1': lambda x: np.exp(-x)}
        exp2 = {'e2': lambda x: np.exp(x)}
        sincos3 = {'g13': lambda x: np.sin(
            2*np.pi*x/T[2]), 'g14': lambda x: np.cos(2*np.pi*x/T[2])}
        sincos4 = {'g13': lambda x: np.sin(
            2*np.pi*x/T[3]), 'g14': lambda x: np.cos(2*np.pi*x/T[3])}
        sincos7 = {'g13': lambda x: np.sin(
            2*np.pi*x/T[6]), 'g14': lambda x: np.cos(2*np.pi*x/T[6])}

        self.basis_functions_dict = {'poly': poly1,
                                     'poly12': poly1 | poly2,
                                     'poly17': poly1 | poly7,
                                     'poly13': poly1 | poly3,
                                     'poly135': poly1 | poly3 | poly5,
                                     'poly1357': poly1 | poly3 | poly5 | poly7,
                                     'poly_sincos4': poly1 | sincos4, 'poly_sincos7': poly1 | sincos7,
                                     'poly_exp1': poly1 | exp1,
                                     'poly_exp2': poly1 | exp2,
                                     'poly_exp12': poly1 | poly2 | exp1,
                                     'poly_exp22': poly1 | poly2 | exp2
                                     }
        if selected_bf_dict is None:
            self.selected_bf_dict = {
                'poly': 1.0,
                'poly17': 1.,
                'poly12': 1.0,
                'poly13': 1.0,
                'poly135': 1.0,
                'poly1357': 1.0
            }
        else:
            self.selected_bf_dict = selected_bf_dict

    def move_column(self, array, from_col, to_col):
        return np.insert(np.delete(array, from_col, axis=1), to_col, array[:, from_col], axis=1)

    def build_sindy_dyn_frcst_inputs(self, z1_k, drivers, X_library_matrix_inputs_norm_dict, X_reg_norm_dict_sindy,
                                     pca_coupling, kp_idx, f10_idx, model_params, normalization_method, input_features, k=0):
        X_k_for_sindy = np.concatenate([z1_k[pca_coupling].reshape(
            (-1, 1)), drivers[f10_idx:, k].reshape((-1, 1))])

        X_library_matrix_inputs_k_norm = u.normalize_with_dict(
            X_k_for_sindy, X_library_matrix_inputs_norm_dict, method=normalization_method)
        X_library_matrix_inputs_k_norm = X_library_matrix_inputs_k_norm/10.

        current_kp = drivers[kp_idx, k]

        library_dict = u.create_library_functions(np.copy(
            X_library_matrix_inputs_k_norm.T), model_params['functions'], input_features)
        theta_k = library_dict['theta'].T

        X_k = np.concatenate(
            [theta_k, np.delete(z1_k, pca_coupling, axis=0)], axis=0)
        X_k_norm = u.normalize_with_dict(
            X_k[1:], X_reg_norm_dict_sindy, method=normalization_method)
        X_k_norm = np.concatenate(
            [X_k[0, :].reshape((1, -1)), drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])

        return X_k_norm

    def ode_func_sindy(self, t, q_norm, drivers, A_low_c, B_low_c, A_mid_c, B_mid_c, A_high_c, B_high_c, sindy_tgt_col_1, pca_couplings, kp_th):

        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1))

        current_kp = drivers[self.kp_idx, int(t)]
        current_f10 = drivers[self.f10_idx, int(t)]

        q_denormalized = u.denormalize(q_norm.reshape(
            (-1, 1)), self.Y_reg_norm_dict_sindy, self.normalization_method)
        X_k_norm = self.build_sindy_dyn_frcst_inputs(q_denormalized, drivers, self.X_library_matrix_inputs_norm_dict,
                                                     self.X_reg_norm_dict_sindy, pca_couplings, self.kp_idx, self.f10_idx, self.model_params,
                                                     self.normalization_method, self.input_features, k=int(t))

        qF_norm = self.move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col_1).T

        F_norm = np.copy(qF_norm[:(-self.n_components), 0]).reshape((-1, 1))
        dq_dt = ((A_low_c * (current_f10 < self.q_low_f10_value) + A_mid_c * ((current_f10 >= self.q_low_f10_value) &
                                                                              (current_f10 < self.q_high_f10_value)) +
                  A_high_c * (current_f10 >= self.q_high_f10_value)) @ q_norm.reshape((-1, 1)) +
                 (B_low_c * (current_f10 < self.q_low_f10_value) + B_mid_c * ((current_f10 >= self.q_low_f10_value) &
                                                                              (current_f10 < self.q_high_f10_value)) +
                  B_high_c * (current_f10 >= self.q_high_f10_value)) @ F_norm.reshape((-1, 1))).flatten()

        return dq_dt

    def ode_func_sindy_with_cov(self, t, y, drivers, A_low_c, B_low_c, A_mid_c, B_mid_c, A_high_c, B_high_c, sindy_tgt_col_1, pca_couplings, kp_th):
        """
        ODE function for state and covariance propagation using the Riccati equation.
        y: concatenated vector of state (q_norm) and flattened covariance (P), shape (n + n*n,)
        Returns: concatenated derivative vector [dq_dt, dP_dt.flatten()]
        """
        n = self.n_components
        q_norm = y[:n]
        P_flat = y[n:]
        P = P_flat.reshape((n, n))

        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1))

        current_kp = drivers[self.kp_idx, int(t)]
        current_f10 = drivers[self.f10_idx, int(t)]

        q_denormalized = u.denormalize(q_norm.reshape(
            (-1, 1)), self.Y_reg_norm_dict_sindy, self.normalization_method)
        X_k_norm = self.build_sindy_dyn_frcst_inputs(q_denormalized, drivers, self.X_library_matrix_inputs_norm_dict,
                                                     self.X_reg_norm_dict_sindy, pca_couplings, self.kp_idx, self.f10_idx, self.model_params,
                                                     self.normalization_method, self.input_features, k=int(t))

        qF_norm = self.move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col_1).T

        F_norm = np.copy(qF_norm[:(-self.n_components), 0]).reshape((-1, 1))
        # Select A and B matrices based on current_f10
        A = (A_low_c * (current_f10 < self.q_low_f10_value) +
             A_mid_c * ((current_f10 >= self.q_low_f10_value) & (current_f10 < self.q_high_f10_value)) +
             A_high_c * (current_f10 >= self.q_high_f10_value))
        B = (B_low_c * (current_f10 < self.q_low_f10_value) +
             B_mid_c * ((current_f10 >= self.q_low_f10_value) & (current_f10 < self.q_high_f10_value)) +
             B_high_c * (current_f10 >= self.q_high_f10_value))

        dq_dt = (A @ q_norm.reshape((-1, 1)) + B @
                 F_norm.reshape((-1, 1))).flatten()

        # Riccati equation: dP/dt = A P + P A^T + Q
        # Q: process noise, set as small diagonal if not provided
        Q = np.eye(n) * 1e-6

        dP_dt = A @ P + P @ A.T + Q

        return np.concatenate([dq_dt, dP_dt.flatten()])

    def ode_func_dmd(self, t, q_norm, drivers, A_c, B_c):
        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1))
        k = int(t)

        q_denormalized = u.denormalize(q_norm.reshape(
            (-1, 1)), self.Y_reg_norm_dict_nl_dmd, self.normalization_method)

        X_k = np.concatenate([q_denormalized, drivers[self.f10_idx:, k].reshape((-1, 1)), (drivers[self.f10_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1)),
                              (drivers[self.kp_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(
            X_k, self.X_reg_norm_dict_nl_dmd, self.normalization_method)
        X_k_norm = np.concatenate(
            [drivers[1:self.f10_idx, k].reshape((-1, 1)), X_k_norm])

        q0_norm = np.copy(X_k_norm[self.time_variables:(-4), :])
        F_norm = np.copy(np.concatenate(
            [X_k_norm[:self.time_variables, :], X_k_norm[(-4):, :]], axis=0))

        dq_dt = (A_c @ q_norm.reshape((-1, 1)) + B_c @
                 F_norm.reshape((-1, 1))).flatten()
        return dq_dt

    def ode_func_dmd_with_cov(self, t, y, drivers, A_c, B_c):
        """
        ODE function for state and covariance propagation using the Riccati equation for DMD.
        y: concatenated vector of state (q_norm) and flattened covariance (P), shape (n + n*n,)
        Returns: concatenated derivative vector [dq_dt, dP_dt.flatten()]
        """
        n = self.n_components
        q_norm = y[:n]
        P_flat = y[n:]
        P = P_flat.reshape((n, n))

        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1))
        k = int(t)

        q_denormalized = u.denormalize(q_norm.reshape(
            (-1, 1)), self.Y_reg_norm_dict_nl_dmd, self.normalization_method)

        X_k = np.concatenate([q_denormalized, drivers[self.f10_idx:, k].reshape((-1, 1)), (drivers[self.f10_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1)),
                              (drivers[self.kp_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(
            X_k, self.X_reg_norm_dict_nl_dmd, self.normalization_method)
        X_k_norm = np.concatenate(
            [drivers[1:self.f10_idx, k].reshape((-1, 1)), X_k_norm])

        q0_norm = np.copy(X_k_norm[self.time_variables:(-4), :])
        F_norm = np.copy(np.concatenate(
            [X_k_norm[:self.time_variables, :], X_k_norm[(-4):, :]], axis=0))

        dq_dt = (A_c @ q_norm.reshape((-1, 1)) + B_c @
                 F_norm.reshape((-1, 1))).flatten()

        # Riccati equation: dP/dt = A P + P A^T + Q
        Q = np.eye(n) * 1e-6

        dP_dt = A_c @ P + P @ A_c.T + Q

        return np.concatenate([dq_dt, dP_dt.flatten()])

    def interpolate_matrix_rows(self, matrix, sub_intervals):
        
        n, m = matrix.shape
        interpolated_columns = m * sub_intervals
        result = np.zeros((n, interpolated_columns))

        for i in range(n):
            x_original = np.arange(m)
            x_interpolated = np.linspace(0, m - 1, interpolated_columns)

            result[i, :] = np.interp(x_interpolated, x_original, matrix[i, :])

        return result

    def find_closest_match(self, df, f10_target, kp_target):
        
        # Compute absolute differences
        df['f10_diff'] = np.abs(df['f10'] - f10_target)
        df['kp_diff'] = np.abs(df['kp'] - kp_target)

        # Find the row with the smallest sum of differences
        closest_row = df.loc[(df['f10_diff'] + df['kp_diff']).idxmin()]
        closest_row = closest_row.drop(columns=['f10_diff', 'kp_diff'])
        # Drop helper columns before returning
        df = df.drop(columns=['f10_diff', 'kp_diff'])

        return closest_row

    def discrete_to_continuous(self, A_d, B_d, delta_t):

        A_c = (1 / delta_t) * logm(A_d)
        B_c = A_c @ (np.linalg.inv(A_d - np.eye(A_d.shape[0])) @ B_d)

        return A_c, B_c

    def get_initial_z_from_drivers(self, sampled_ic_table, f10_input, kp_input):

        f10_sorted = np.sort(sampled_ic_table['f10'].unique())
        # if any(f10_sorted <= f10_input) #else f10_sorted[f10_sorted <= f10_input].max()
        f10_below = f10_sorted[f10_sorted <= f10_input].max()
        kp_sorted = np.sort(sampled_ic_table['kp'].unique())
        # if any(kp_sorted <= kp_input) else kp_sorted[kp_sorted <= kp_input].max()
        kp_below = kp_sorted[kp_sorted <= kp_input].max()
        if f10_below is None or kp_below is None:
            return None

        row = self.find_closest_match(sampled_ic_table, f10_below, kp_below)

        return row.iloc[2:12].squeeze() if not row.empty else None
