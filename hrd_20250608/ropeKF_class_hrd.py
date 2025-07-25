import numpy as np
from os import path
import numpy as np
import pandas as pd
from scipy.linalg import logm
from scipy.integrate import solve_ivp
from datetime import datetime
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

    def propagate_models_mins(self, start_time: datetime, end_time: datetime):
        """
        Propagate the models over the given time interval.

        Parameters:
        -----------
        start_time : datetime
            The start time of the propagation.
        end_time : datetime
            The end time of the propagation.

        Returns:
        --------
        None
        """

        propagation_time_min = (end_time - start_time).total_seconds() / 60
        prop_year = start_time.year
        self.date_series = pd.date_range(
            start=start_time, end=end_time, freq=f'{self.sub_intervals}s', inclusive='left')
        self.hourly_date_series = pd.date_range(
            start=start_time, end=end_time, freq='H', inclusive='left')

        n_components = self.n_components
        normalization_method = 'std'
        gamma = 1
        delta_t = gamma*self.sub_intervals
        time_offset = self.delta_rho_ic

        start_of_year = datetime(start_time.year, 1, 1)
        t0 = (start_time - start_of_year).total_seconds() / 60  # mins
        tf = t0 + propagation_time_min  # mins

        f10_idx = self.f10_idx
        kp_idx = self.kp_idx
        input_data_models = self.input_data_sindy['models_coefficients'][()]
        B_nl_dmd_discrete = input_data_models['models_dict']['nl-dmd']['plain']['ridge_parameter_1.00']

        IC_idx_at_start_of_year = np.where(self.drivers[0, :] == prop_year)[0]
        start_idx = int(np.min(IC_idx_at_start_of_year) - time_offset)  # hours
        end_idx = int(np.min(IC_idx_at_start_of_year) +
                      int(np.ceil(tf / 60)))  # hours
        indices_with_time_offset = np.arange(start_idx, end_idx)

        drivers_at_toffset = np.copy(self.drivers[:, indices_with_time_offset])
        self.drivers_IC = drivers_at_toffset

        t0_hr = int(t0 // 60)
        tf_hr = int(np.ceil(tf / 60))
        drivers = np.copy(drivers_at_toffset[:, t0_hr:tf_hr])

        interpolated_drivers = self.interpolate_matrix_rows(
            drivers, self.sub_intervals)  # back in mins
        self.interval_interpolated_drivers = interpolated_drivers[:, int(
            # in mins
            self.sub_intervals*self.delta_rho_ic): int(propagation_time_min)]
        self.interval_hourly_drivers = drivers[:, int(
            time_offset):(drivers.shape[1])]

        # if not hasattr(self, 'propagation_drivers'):
        self.t0 = t0
        self.tf = tf
        self.propagation_drivers = interpolated_drivers

        f10_value = np.copy(drivers_at_toffset[f10_idx, t0_hr])
        kp_value = np.copy(drivers_at_toffset[kp_idx, t0_hr])
        z_series = self.get_initial_z_from_drivers(
            self.initial_conditions, f10_value, kp_value)
        z1_k = z_series.values.reshape((self.n_components, 1))
        input_features = ['x_' + str(k+1).zfill(2)
                          for k in range(z1_k.shape[0])]

        k = 0
        # Start and end times for ivp integration in mins
        t_span = (0, int(propagation_time_min))
        # time points at which to evaluate the solution in mins
        t_interval = np.linspace(
            t_span[0], t_span[1], ((t_span[1] - t_span[0]) + 1))
        self.t_interval = t_interval[self.sub_intervals*self.delta_rho_ic:]
        self.z_results_lst = []

        for chosen_basis_function in list(self.selected_bf_dict.keys()):
            warnings.filterwarnings("ignore")
            ridge_label = 'ridge_parameter_' + \
                "{:.2f}".format(self.selected_bf_dict[chosen_basis_function])

            self.B_sindy_joint_discrete = input_data_models['models_dict'][
                'sindy'][chosen_basis_function][ridge_label]['joint']
            self.B_sindy_f10_discrete = input_data_models['models_dict'][
                'sindy'][chosen_basis_function][ridge_label]['sm_f10']
            self.B_sindy_combined_discrete = input_data_models['models_dict'][
                'sindy'][chosen_basis_function][ridge_label]['combined']
            self.B_sindy_joint_low_d = input_data_models['models_dict'][
                'sindy'][chosen_basis_function][ridge_label]['joint_low']
            self.B_sindy_joint_mid_d = input_data_models['models_dict'][
                'sindy'][chosen_basis_function][ridge_label]['joint_mid']
            self.B_sindy_joint_high_d = input_data_models['models_dict'][
                'sindy'][chosen_basis_function][ridge_label]['joint_high']

            self.X_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][
                chosen_basis_function]['X_reg_norm_dict_sindy']
            self.Y_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][
                chosen_basis_function]['Y_reg_norm_dict_sindy']
            self.X_library_matrix_inputs_norm_dict = input_data_models['bf_normalization_dict'][
                chosen_basis_function]['X_library_matrix_inputs_norm_dict']

            self.model_params = {'normalization_method': 'std',
                                 'functions': self.basis_functions_dict[chosen_basis_function]}
            self.sindy_tgt_col = self.B_sindy_joint_discrete.shape[1] - \
                n_components + self.pca_coupling

            ########################################### sindy###############################################
            array_joint = self.move_column(
                np.copy(self.B_sindy_joint_discrete), 5, self.sindy_tgt_col)
            A_sindy_joint = np.copy(array_joint[:, -n_components:])
            B_sindy_joint = np.copy(array_joint[:, :(-n_components)])
            A_sindy_joint_c, B_sindy_joint_c = self.discrete_to_continuous(
                A_sindy_joint, B_sindy_joint, delta_t)

            array_combined = self.move_column(
                np.copy(self.B_sindy_combined_discrete), 5, self.sindy_tgt_col)
            A_sindy_combined = np.copy(array_combined[:, -n_components:])
            B_sindy_combined = np.copy(array_combined[:, :(-n_components)])
            A_sindy_combined_c, B_sindy_combined_c = self.discrete_to_continuous(
                A_sindy_combined, B_sindy_combined, delta_t)

            array_f10 = self.move_column(
                np.copy(self.B_sindy_f10_discrete), 5, self.sindy_tgt_col)
            A_sindy_f10 = np.copy(array_f10[:, -n_components:])
            B_sindy_f10 = np.copy(array_f10[:, :(-n_components)])
            A_sindy_f10_c, B_sindy_f10_c = self.discrete_to_continuous(
                A_sindy_f10, B_sindy_f10, delta_t)

            array_joint_low = self.move_column(
                np.copy(self.B_sindy_joint_low_d), 5, self.sindy_tgt_col)
            A_sindy_joint_low = np.copy(array_joint_low[:, -n_components:])
            B_sindy_joint_low = np.copy(array_joint_low[:, :(-n_components)])
            A_sindy_joint_low_c, B_sindy_joint_low_c = self.discrete_to_continuous(
                A_sindy_joint_low, B_sindy_joint_low, delta_t)

            array_joint_mid = self.move_column(
                np.copy(self.B_sindy_joint_mid_d), 5, self.sindy_tgt_col)
            A_sindy_joint_mid = np.copy(array_joint_mid[:, -n_components:])
            B_sindy_joint_mid = np.copy(array_joint_mid[:, :(-n_components)])
            A_sindy_joint_mid_c, B_sindy_joint_mid_c = self.discrete_to_continuous(
                A_sindy_joint_mid, B_sindy_joint_mid, delta_t)

            array_joint_high = self.move_column(
                np.copy(self.B_sindy_joint_high_d), 5, self.sindy_tgt_col)
            A_sindy_joint_high = np.copy(array_joint_high[:, -n_components:])
            B_sindy_joint_high = np.copy(array_joint_high[:, :(-n_components)])
            A_sindy_joint_high_c, B_sindy_joint_high_c = self.discrete_to_continuous(
                A_sindy_joint_high, B_sindy_joint_high, delta_t)

            X_k_norm = self.build_sindy_dyn_frcst_inputs(z1_k, interpolated_drivers,
                                                         self.X_library_matrix_inputs_norm_dict, self.X_reg_norm_dict_sindy, self.pca_coupling,
                                                         kp_idx, f10_idx, self.model_params, normalization_method, input_features, k=int(k))

            qF_norm = self.move_column(
                np.copy(X_k_norm).T, 5, self.sindy_tgt_col).T
            q0_norm_sindy = np.copy(qF_norm[-n_components:])
            solution_sindy = solve_ivp(
                self.ode_func_sindy,
                t_span,
                q0_norm_sindy.flatten(),
                args=(interpolated_drivers, A_sindy_joint_low_c, B_sindy_joint_low_c, A_sindy_joint_mid_c,
                      B_sindy_joint_mid_c, A_sindy_joint_high_c, B_sindy_joint_high_c, self.sindy_tgt_col, self.pca_coupling,
                      self.kp_th),
                method='RK45',
                t_eval=t_interval
            )

            t = solution_sindy.t

            q_sol_sindy = np.full(
                (len(q0_norm_sindy.flatten()), len(t_interval)), np.nan)
            q_sol_sindy[:, :len(solution_sindy.t)] = solution_sindy.y
            z_sindy = np.copy(
                q_sol_sindy * self.Y_reg_norm_dict_sindy['x_std'] + self.Y_reg_norm_dict_sindy['x_mean'])
            z_sindy = z_sindy[:, int(self.sub_intervals*self.delta_rho_ic):]
            self.z_results_lst.append(z_sindy)

        A_dmd = np.copy(
            np.copy(B_nl_dmd_discrete[:, self.time_variables:(-4)]))
        B_dmd = np.copy(np.concatenate(
            [B_nl_dmd_discrete[:, :self.time_variables], B_nl_dmd_discrete[:, (-4):]], axis=1))

        A_dmd_c, B_dmd_c = self.discrete_to_continuous(A_dmd, B_dmd, delta_t)

        X_k = np.concatenate([z1_k, interpolated_drivers[f10_idx:, k].reshape((-1, 1)), (interpolated_drivers[f10_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1)),
                              (interpolated_drivers[kp_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(
            X_k, self.X_reg_norm_dict_nl_dmd, normalization_method)
        X_k_norm = np.concatenate(
            [interpolated_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
        q0_norm_dmd = np.copy(X_k_norm[self.time_variables:(-4), :])

        solution_dmd = solve_ivp(
            self.ode_func_dmd,
            t_span,
            q0_norm_dmd.flatten(),
            args=(interpolated_drivers, A_dmd_c, B_dmd_c),
            method='RK45',
            t_eval=t_interval
        )

        t = solution_dmd.t

        q_sol_dmd = np.full(
            (len(q0_norm_dmd.flatten()), len(t_interval)), np.nan)
        q_sol_dmd[:, :len(solution_dmd.t)] = solution_dmd .y
        z_dmd = np.copy(
            q_sol_dmd * self.Y_reg_norm_dict_nl_dmd['x_std'] + self.Y_reg_norm_dict_nl_dmd['x_mean'])
        z_dmd = z_dmd[:, self.sub_intervals*self.delta_rho_ic:]
        self.z_results_lst.append(z_dmd)

        if self.delta_rho_ic != 0:
            self.t = t[:(-int(self.sub_intervals*self.delta_rho_ic))]
        else:
            self.t = t
        self.z_dict = {}
        models = list(self.selected_bf_dict.keys()) + ['dmd']
        for k, z in enumerate(self.z_results_lst):
            self.z_dict[models[k]] = self.z_results_lst[k]
