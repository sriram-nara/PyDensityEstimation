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

class rope_propagator:
    '''
    Class to store and propagate tyhe SINDY models based on TIEGCM physics based dataset
    '''
    def __init__( self, datapath: str = "hrd_20250608", drivers = None, selected_bf_dict = None, delta_rho_ic = 0):
        #72: longitude intervals, 36: latitude intervals, 45: altitude intervals
        self.input_data_sindy = np.load(path.join(datapath, 'z_drivers_dataset_hrd_02_09_std_rescaling_v13.npz'), allow_pickle=True)
        self.U0 = self.input_data_sindy['u_svd'].reshape((72, 36, 45, 10), order='C')
        self.mu0 = self.input_data_sindy['mu_svd'].reshape((72, 36, 45), order='C')
        self.models_coefficients = self.input_data_sindy['models_coefficients']
        self.initial_conditions = pd.DataFrame(self.input_data_sindy['initial_conditions'][()], columns = [ 'f10', 'kp']+[f'z_{str(k).zfill(2)}' for k in range(10)])
        if drivers is None:
            self.drivers = self.input_data_sindy['celestrack_drivers']
        else:
            self.drivers = drivers
            self.original_drivers = self.input_data_sindy['celestrack_drivers']
    
        self.X_reg_norm_dict_nl_dmd = self.input_data_sindy['models_coefficients'][()]['X_reg_norm_dict_nl_dmd']
        self.Y_reg_norm_dict_nl_dmd = self.input_data_sindy['models_coefficients'][()]['Y_reg_norm_dict_nl_dmd']

        self.q_low_f10_value = 100.#self.input_data_sindy['models_coefficients'][()]['q_low_f10_value']
        self.q_high_f10_value = 160.#self.input_data_sindy['models_coefficients'][()]['q_high_f10_value']

        self.x_train_svd_obj = SvdContainer(self.input_data_sindy['u_svd'], self.input_data_sindy['mu_svd'])
        self.x_train_svd_obj.norm_dict = {'x_mean': self.input_data_sindy['mu_svd']}
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
        self.input_features = ['x_'+ str(k+1).zfill(2) for k in range(10)]
        poly1 = {'p1': lambda x: x}
        poly1p5 = {'p1': lambda x: np.abs(x)**1.5}
        poly2 = {'p2': lambda x: x**2}
        poly3 = {'p3': lambda x: x**3}
        poly4 = {'p3': lambda x: x**4}
        poly5 = {'p5': lambda x: x**5}
        poly7 = {'p5': lambda x: x**7}
        exp1 = {'e1': lambda x: np.exp(-x)}
        exp2 = {'e2': lambda x: np.exp(x)}
        sincos3 = {'g13': lambda x: np.sin(2*np.pi*x/T[2]), 'g14': lambda x: np.cos(2*np.pi*x/T[2])}
        sincos4 = {'g13': lambda x: np.sin(2*np.pi*x/T[3]), 'g14': lambda x: np.cos(2*np.pi*x/T[3])}
        sincos7 = {'g13': lambda x: np.sin(2*np.pi*x/T[6]), 'g14': lambda x: np.cos(2*np.pi*x/T[6])}

        self.basis_functions_dict = {'poly':poly1, 
                        'poly12': poly1 | poly2, 
                        'poly17':poly1|poly7,
                        'poly13':poly1|poly3,
                        'poly135':poly1|poly3|poly5,
                        'poly1357':poly1|poly3|poly5|poly7,
                        'poly_sincos4': poly1 | sincos4, 'poly_sincos7': poly1 | sincos7,\
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
        
    
    def build_sindy_dyn_frcst_inputs(self, z1_k, drivers, X_library_matrix_inputs_norm_dict, X_reg_norm_dict_sindy, pca_coupling, kp_idx, f10_idx, model_params, normalization_method, input_features, k = 0):
        X_k_for_sindy = np.concatenate([z1_k[pca_coupling].reshape((-1, 1)), drivers[f10_idx:, k].reshape((-1, 1))])
        
        X_library_matrix_inputs_k_norm = u.normalize_with_dict(X_k_for_sindy, X_library_matrix_inputs_norm_dict, method = normalization_method) 
        X_library_matrix_inputs_k_norm = X_library_matrix_inputs_k_norm/10.
        # X_library_matrix_inputs_k_norm = X_k_for_sindy/10.
        # iq25 = np.quantile(X_k_for_sindy, 0.25, axis = 0)
        # iq50 = np.quantile(X_k_for_sindy, 0.50, axis = 0)
        # iq75 = np.quantile(X_k_for_sindy, 0.75, axis = 0)
        # X_library_matrix_inputs_k_norm = np.copy((X_k_for_sindy)/(iq75 - iq25))

        current_kp = drivers[kp_idx, k]

        
        library_dict = u.create_library_functions(np.copy(X_library_matrix_inputs_k_norm.T), model_params['functions'], input_features)
        theta_k = library_dict['theta'].T
        
        X_k = np.concatenate([theta_k, np.delete(z1_k, pca_coupling, axis = 0)], axis = 0)     
        X_k_norm = u.normalize_with_dict(X_k[1:], X_reg_norm_dict_sindy, method = normalization_method)  
        X_k_norm = np.concatenate([X_k[0, :].reshape((1, -1)), drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm]) 

        return X_k_norm

    # def build_sindy_dyn_frcst_inputs(self, z1_k, drivers, X_reg_norm_dict_sindy, pca_coupling, kp_idx, f10_idx, model_params, normalization_method, input_features, k = 0):
    #     X_k_for_sindy = np.concatenate([z1_k[pca_coupling].reshape((-1, 1)), drivers[f10_idx:, k].reshape((-1, 1))])
    #     library_dict = u.create_library_functions(np.copy(X_k_for_sindy.T), model_params['functions'], input_features)
    #     theta_k = library_dict['theta'].T
    #     X_k = np.concatenate([theta_k, np.delete(z1_k, pca_coupling, axis = 0)], axis = 0)     
    #     X_k_norm = u.normalize_with_dict(X_k[1:], X_reg_norm_dict_sindy, method = normalization_method)  
    #     X_k_norm = np.concatenate([X_k[0, :].reshape((1, -1)), drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm]) 

    #     return X_k_norm

    
    def ode_func_sindy(self, t, q_norm, drivers, A_low_c, B_low_c, A_mid_c, B_mid_c, A_high_c, B_high_c, sindy_tgt_col_1, pca_couplings, kp_th):

        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1)) 
        
        current_kp = drivers[self.kp_idx, int(t)]
        current_f10 = drivers[self.f10_idx, int(t)]
        
        q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), self.Y_reg_norm_dict_sindy, self.normalization_method)
        X_k_norm = self.build_sindy_dyn_frcst_inputs(q_denormalized, drivers, self.X_library_matrix_inputs_norm_dict, \
            self.X_reg_norm_dict_sindy, pca_couplings, self.kp_idx, self.f10_idx, self.model_params,\
                self.normalization_method, self.input_features, k = int(t)) 
        # X_k_norm = self.build_sindy_dyn_frcst_inputs(q_denormalized, drivers, self.X_reg_norm_dict_sindy, pca_couplings, self.kp_idx, self.f10_idx, self.model_params,\
        #     self.normalization_method, self.input_features, k = int(t)) 
        
        qF_norm = self.move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col_1).T
        
        F_norm = np.copy(qF_norm[:(-self.n_components), 0]).reshape((-1, 1))
        # print((current_f10 < self.q_low_f10_value), (current_f10 >= self.q_low_f10_value) & (current_f10 < self.q_high_f10_value), (current_f10 >= self.q_high_f10_value))
        dq_dt = ( ( A_low_c * (current_f10 < self.q_low_f10_value) + A_mid_c * ((current_f10 >= self.q_low_f10_value) & (current_f10 < self.q_high_f10_value)) + A_high_c * (current_f10 >= self.q_high_f10_value) ) @ q_norm.reshape((-1, 1)) + \
            (B_low_c * (current_f10 < self.q_low_f10_value) + B_mid_c * ((current_f10 >= self.q_low_f10_value) & (current_f10 < self.q_high_f10_value)) + B_high_c * (current_f10 >= self.q_high_f10_value)) @ F_norm.reshape((-1, 1)) ).flatten()   
        # dq_dt = ( ( A_low_c * (current_kp < 5.) + A_mid_c * (current_kp >= 5.)) @ q_norm.reshape((-1, 1)) + \
        #     (B_low_c * (current_kp < 5.) + B_mid_c * (current_kp >= 5.) ) @ F_norm.reshape((-1, 1)) ).flatten()   
        # print(dq_dt)
        # nansum = np.isnan(dq_dt).sum()
        # print(np.any(dq_dt > 1000.))
        # if (np.any(dq_dt > 500.)) | nansum:
        #     # print('Emergency routine 1')
        #     # print(dq_dt)
        #     z_series = self.get_initial_z_from_drivers(self.initial_conditions, current_f10, current_kp)
        #     z1_k = z_series.values.reshape((self.n_components, 1))
        #     X_k_norm = self.build_sindy_dyn_frcst_inputs(z1_k, drivers, \
        #         self.X_library_matrix_inputs_norm_dict, self.X_reg_norm_dict_sindy, self.pca_coupling, \
        #             self.kp_idx, self.f10_idx, self.model_params, self.normalization_method, self.input_features, k = int(t))
        #     qF_norm = self.move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col_1).T
        #     F_norm = np.copy(qF_norm[:(-self.n_components), 0]).reshape((-1, 1))
        #     q_norm = np.copy(qF_norm[-self.n_components:])

        #     # dq_dt = ( ( A_low_c * (current_f10 < self.q_low_f10_value) + A_mid_c * ((current_f10 >= self.q_low_f10_value) & (current_f10 < self.q_high_f10_value)) + A_high_c * (current_f10 >= self.q_high_f10_value) ) @ q_norm.reshape((-1, 1)) + \
        #     #             (B_low_c * (current_f10 < self.q_low_f10_value) + B_mid_c * ((current_f10 >= self.q_low_f10_value) & (current_f10 < self.q_high_f10_value)) + B_high_c * (current_f10 >= self.q_high_f10_value)) @ F_norm.reshape((-1, 1)) ).flatten()   
        #     return q_norm.flatten()
               
        return dq_dt


    # def ode_func_sindy(self, t, q_norm, drivers, A_sindy_joint_c, B_sindy_joint_c, A_sindy_f10_c, B_sindy_f10_c, sindy_tgt_col_1, pca_couplings, kp_th):
    #     discrete_idx = np.searchsorted(self.t_interval, t)
    #     discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1)) 
        
    #     current_kp = drivers[self.kp_idx, int(t)]
        
    #     q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), self.Y_reg_norm_dict_sindy, self.normalization_method)
    #     X_k_norm = self.build_sindy_dyn_frcst_inputs(q_denormalized, drivers, self.X_library_matrix_inputs_norm_dict, self.X_reg_norm_dict_sindy, pca_couplings, self.kp_idx, self.f10_idx, self.model_params,\
    #         self.normalization_method, self.input_features, k = int(t)) 
    #     # X_k_norm = self.build_sindy_dyn_frcst_inputs(q_denormalized, drivers, self.X_reg_norm_dict_sindy, pca_couplings, self.kp_idx, self.f10_idx, self.model_params,\
    #     #     self.normalization_method, self.input_features, k = int(t)) 
        
    #     qF_norm = self.move_column(np.copy(X_k_norm).T, 5, sindy_tgt_col_1).T  
        
    #     F_norm = np.copy(qF_norm[:(-self.n_components), 0]).reshape((-1, 1)) 
    #     dq_dt = ( (A_sindy_joint_c * (current_kp >= kp_th) + A_sindy_f10_c * (current_kp < kp_th)) @ q_norm.reshape((-1, 1)) + \
    #             (B_sindy_joint_c * (current_kp >= kp_th) + B_sindy_f10_c * (current_kp < kp_th)) @ F_norm.reshape((-1, 1)) ).flatten()   
    #     return dq_dt

    def ode_func_dmd(self, t, q_norm, drivers, A_c, B_c):
        discrete_idx = np.searchsorted(self.t_interval, t)
        discrete_idx = max(0, min(int(discrete_idx), len(self.t_interval) - 1)) 
        k = int(t)
        
        q_denormalized = u.denormalize(q_norm.reshape((-1, 1)), self.Y_reg_norm_dict_nl_dmd, self.normalization_method)
        
        X_k = np.concatenate([q_denormalized, drivers[self.f10_idx:, k].reshape((-1, 1)), (drivers[self.f10_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1)), 
            (drivers[self.kp_idx, k] * drivers[self.kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(X_k, self.X_reg_norm_dict_nl_dmd, self.normalization_method)
        X_k_norm = np.concatenate([drivers[1:self.f10_idx, k].reshape((-1, 1)), X_k_norm])
        
        q0_norm = np.copy(X_k_norm[self.time_variables:(-4), :])
        F_norm = np.copy(np.concatenate([X_k_norm[:self.time_variables, :], X_k_norm[(-4):, :]], axis = 0))
        
        dq_dt = (A_c @ q_norm.reshape((-1, 1)) + B_c @ F_norm.reshape((-1, 1))).flatten()   
        return dq_dt

    
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
        f10_below = f10_sorted[f10_sorted <= f10_input].max() #if any(f10_sorted <= f10_input) #else f10_sorted[f10_sorted <= f10_input].max()
        kp_sorted = np.sort(sampled_ic_table['kp'].unique())
        kp_below = kp_sorted[kp_sorted <= kp_input].max() #if any(kp_sorted <= kp_input) else kp_sorted[kp_sorted <= kp_input].max()
        if f10_below is None or kp_below is None:
            return None
        
        row = self.find_closest_match(sampled_ic_table, f10_below, kp_below)
        
        return row.iloc[2:12].squeeze() if not row.empty else None

    def propagate_models(self, init_date, forward_propagation = 5):

        init_date = pd.to_datetime(init_date)
        year = init_date.year
        day_of_year = init_date.day_of_year
        hour0 = init_date.hour

        start_date = init_date
        end_date = datetime(year, 1, 1) + timedelta(days=day_of_year + forward_propagation - 1)
    

        # Generate date series with sub_intervals-seconds resolution for n days
        self.date_series = pd.date_range(start=start_date, end = end_date, freq = str(self.sub_intervals) + 's')[:(-1)]
        self.hourly_date_series = pd.date_range(start=start_date, end = end_date, freq = 'H')[:(-1)]

        n_components = self.n_components
        normalization_method = 'std'
        gamma = 1
        delta_t = gamma*self.sub_intervals 
        time_offset = self.delta_rho_ic
        t0 = day_of_year * 24 - 24 + hour0# 24 hours of day doy have not passed yet
        n_days_frcst = forward_propagation
        forward_hours = n_days_frcst * 24
        tf = t0 + forward_hours + self.delta_rho_ic - hour0 
        f10_idx = self.f10_idx
        kp_idx = self.kp_idx
        input_data_models = self.input_data_sindy['models_coefficients'][()]
        B_nl_dmd_discrete = input_data_models['models_dict']['nl-dmd']['plain']['ridge_parameter_1.00']

        IC_idx_at_start_of_year = np.where(self.drivers[0,:] == year)[0]
        indices_with_time_offset = np.arange(np.min(IC_idx_at_start_of_year) - time_offset, np.min(IC_idx_at_start_of_year) + tf)
        drivers_at_toffset = np.copy(self.drivers[:, indices_with_time_offset]) 
        self.drivers_IC = drivers_at_toffset        
        drivers = np.copy(drivers_at_toffset[:, (t0):(tf)])
        

        interpolated_drivers = self.interpolate_matrix_rows(drivers, self.sub_intervals)
        self.interval_interpolated_drivers = interpolated_drivers[:, int(self.sub_intervals*self.delta_rho_ic):int(self.sub_intervals*drivers.shape[1])]
        self.interval_hourly_drivers = drivers[:, int(time_offset):(drivers.shape[1])]

        if not hasattr(self, 'propagation_drivers'):
            self.t0 = t0
            self.tf = tf
            self.propagation_drivers = interpolated_drivers

        f10_value = np.copy(drivers_at_toffset[f10_idx, t0])
        kp_value = np.copy(drivers_at_toffset[kp_idx, t0])
        z_series = self.get_initial_z_from_drivers(self.initial_conditions, f10_value, kp_value)
        z1_k = z_series.values.reshape((self.n_components, 1))
        input_features = ['x_'+ str(k+1).zfill(2) for k in range(z1_k.shape[0])]
        k = 0
        t_span = (0, self.sub_intervals*(tuple(range(drivers.shape[1]))[-1] + 1) - 1) #Start and end times for ivp integration
        t_interval = np.linspace(t_span[0], t_span[1], ((t_span[1] - t_span[0]) + 1)) #time points at which to evaluate the solution
        self.t_interval = t_interval[self.sub_intervals*self.delta_rho_ic:]

        print(f'Maximum available time T = {(self.sub_intervals*(24*n_days_frcst - hour0)-1)*60}s')
        self.z_results_lst = []

        for chosen_basis_function in list(self.selected_bf_dict.keys()):
            warnings.filterwarnings("ignore")
            ridge_label = 'ridge_parameter_' + "{:.2f}".format(self.selected_bf_dict[chosen_basis_function])

            self.B_sindy_joint_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint']
            self.B_sindy_f10_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['sm_f10']
            self.B_sindy_combined_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['combined']
            self.B_sindy_joint_low_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_low']
            self.B_sindy_joint_mid_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_mid']
            self.B_sindy_joint_high_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_high']


            self.X_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['X_reg_norm_dict_sindy']
            self.Y_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['Y_reg_norm_dict_sindy']
            self.X_library_matrix_inputs_norm_dict = input_data_models['bf_normalization_dict'][chosen_basis_function]['X_library_matrix_inputs_norm_dict']
            
            self.model_params = {'normalization_method': 'std', 'functions': self.basis_functions_dict[chosen_basis_function]}
            self.sindy_tgt_col = self.B_sindy_joint_discrete.shape[1] - n_components + self.pca_coupling
            
            ###########################################sindy###############################################
            array_joint = self.move_column(np.copy(self.B_sindy_joint_discrete), 5, self.sindy_tgt_col)
            A_sindy_joint = np.copy(array_joint[:, -n_components:])
            B_sindy_joint = np.copy(array_joint[:, :(-n_components)])
            A_sindy_joint_c, B_sindy_joint_c = self.discrete_to_continuous(A_sindy_joint, B_sindy_joint, delta_t)

            array_combined = self.move_column(np.copy(self.B_sindy_combined_discrete), 5, self.sindy_tgt_col)
            A_sindy_combined = np.copy(array_combined[:, -n_components:])
            B_sindy_combined = np.copy(array_combined[:, :(-n_components)])
            A_sindy_combined_c, B_sindy_combined_c = self.discrete_to_continuous(A_sindy_combined, B_sindy_combined, delta_t)

            array_f10 = self.move_column(np.copy(self.B_sindy_f10_discrete), 5, self.sindy_tgt_col)
            A_sindy_f10 = np.copy(array_f10[:, -n_components:])
            B_sindy_f10 = np.copy(array_f10[:, :(-n_components)])
            A_sindy_f10_c, B_sindy_f10_c = self.discrete_to_continuous(A_sindy_f10, B_sindy_f10, delta_t)   

            array_joint_low = self.move_column(np.copy(self.B_sindy_joint_low_d), 5, self.sindy_tgt_col)
            A_sindy_joint_low = np.copy(array_joint_low[:, -n_components:])
            B_sindy_joint_low = np.copy(array_joint_low[:, :(-n_components)])
            A_sindy_joint_low_c, B_sindy_joint_low_c = self.discrete_to_continuous(A_sindy_joint_low, B_sindy_joint_low, delta_t)

            array_joint_mid = self.move_column(np.copy(self.B_sindy_joint_mid_d), 5, self.sindy_tgt_col)
            A_sindy_joint_mid = np.copy(array_joint_mid[:, -n_components:])
            B_sindy_joint_mid = np.copy(array_joint_mid[:, :(-n_components)])
            A_sindy_joint_mid_c, B_sindy_joint_mid_c = self.discrete_to_continuous(A_sindy_joint_mid, B_sindy_joint_mid, delta_t)

            array_joint_high = self.move_column(np.copy(self.B_sindy_joint_high_d), 5, self.sindy_tgt_col)
            A_sindy_joint_high = np.copy(array_joint_high[:, -n_components:])
            B_sindy_joint_high = np.copy(array_joint_high[:, :(-n_components)])
            A_sindy_joint_high_c, B_sindy_joint_high_c = self.discrete_to_continuous(A_sindy_joint_high, B_sindy_joint_high, delta_t)
            
  
            X_k_norm = self.build_sindy_dyn_frcst_inputs(z1_k, interpolated_drivers, \
                self.X_library_matrix_inputs_norm_dict, self.X_reg_norm_dict_sindy, self.pca_coupling, \
                    kp_idx, f10_idx, self.model_params, normalization_method, input_features, k = int(k))   
            

            
            qF_norm = self.move_column(np.copy(X_k_norm).T, 5, self.sindy_tgt_col).T
            q0_norm_sindy = np.copy(qF_norm[-n_components:])
            solution_sindy = solve_ivp(
                self.ode_func_sindy,
                t_span,
                q0_norm_sindy.flatten(),
                args = (interpolated_drivers, A_sindy_joint_low_c, B_sindy_joint_low_c, A_sindy_joint_mid_c, \
                    B_sindy_joint_mid_c, A_sindy_joint_high_c, B_sindy_joint_high_c, self.sindy_tgt_col, self.pca_coupling, \
                        self.kp_th),
                method = 'RK45',
                t_eval = t_interval
            )
    
            t = self.sub_intervals*solution_sindy.t

            q_sol_sindy = np.full((len(q0_norm_sindy.flatten()), len(t_interval)), np.nan)
            q_sol_sindy[:, :len(solution_sindy.t)] = solution_sindy.y
            z_sindy = np.copy(q_sol_sindy * self.Y_reg_norm_dict_sindy['x_std'] + self.Y_reg_norm_dict_sindy['x_mean'])
            z_sindy = z_sindy[:, int(self.sub_intervals*self.delta_rho_ic):]
            self.z_results_lst.append(z_sindy)
        
        ###########################################dmd###############################################

        A_dmd = np.copy(np.copy(B_nl_dmd_discrete[:, self.time_variables:(-4)]))
        B_dmd = np.copy(np.concatenate([B_nl_dmd_discrete[:, :self.time_variables], B_nl_dmd_discrete[:, (-4):]], axis = 1))

        A_dmd_c, B_dmd_c = self.discrete_to_continuous(A_dmd, B_dmd, delta_t)


        X_k = np.concatenate([z1_k, interpolated_drivers[f10_idx:, k].reshape((-1, 1)), (interpolated_drivers[f10_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1)), 
            (interpolated_drivers[kp_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(X_k, self.X_reg_norm_dict_nl_dmd, normalization_method)   
        X_k_norm = np.concatenate([interpolated_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
        q0_norm_dmd = np.copy(X_k_norm[self.time_variables:(-4), :])

        solution_dmd = solve_ivp(
            self.ode_func_dmd,
            t_span,
            q0_norm_dmd.flatten(),
            args = (interpolated_drivers, A_dmd_c, B_dmd_c),
            method = 'RK45',
            t_eval = t_interval
        )
        
        t = self.sub_intervals*solution_dmd.t


        q_sol_dmd = np.full((len(q0_norm_dmd.flatten()), len(t_interval)), np.nan)
        q_sol_dmd[:, :len(solution_dmd.t)] = solution_dmd .y
        z_dmd = np.copy(q_sol_dmd * self.Y_reg_norm_dict_nl_dmd['x_std'] + self.Y_reg_norm_dict_nl_dmd['x_mean'])
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


    def propagate_models_mins(self, init_date, forward_propagation = 1):

        init_date = pd.to_datetime(init_date)
        year = init_date.year
        day_of_year = init_date.day_of_year
        hour0 = init_date.hour

        start_date = init_date
        # print(start_date)
        end_date = init_date + timedelta(minutes=forward_propagation)
        # print(end_date)

        # Generate date series with sub_intervals-seconds resolution for n days
        self.date_series = pd.date_range(start=start_date, end = end_date, freq = str(self.sub_intervals) + 's')[:(-1)]
        self.hourly_date_series = pd.date_range(start=start_date, end = end_date, freq = 'H')[:(-1)]

        n_components = self.n_components
        normalization_method = 'std'
        gamma = 1
        delta_t = gamma*self.sub_intervals 
        time_offset = self.delta_rho_ic
        t0 = day_of_year * 24 - 24 + hour0 # 24 hours of day have not passed yet

        n_mins_frcst = forward_propagation
        forward_hours = n_mins_frcst / 60

        # If less than 1 day, handle as special case
        if forward_hours < 24:
            # Use only the required number of minutes/hours for drivers
            f10_idx = self.f10_idx
            kp_idx = self.kp_idx
            input_data_models = self.input_data_sindy['models_coefficients'][()]
            B_nl_dmd_discrete = input_data_models['models_dict']['nl-dmd']['plain']['ridge_parameter_1.00']

            IC_idx_at_start_of_year = np.where(self.drivers[0,:] == year)[0]
            # tf must be int for indexing
            # tf = int(np.ceil(t0 + forward_hours + self.delta_rho_ic - hour0))

            
            tf = int(np.ceil(t0 + forward_hours))

            if tf == t0:
                tf += 1  # Ensure at least one step for integration

            indices_with_time_offset = np.arange(np.min(IC_idx_at_start_of_year) - time_offset, np.min(IC_idx_at_start_of_year) + tf)

            drivers_at_toffset = np.copy(self.drivers[:, indices_with_time_offset]) 
            self.drivers_IC = drivers_at_toffset        
            drivers = np.copy(drivers_at_toffset[:, (t0):(tf)])

            # print(t0,tf)

            interpolated_drivers = self.interpolate_matrix_rows(drivers, self.sub_intervals)
            self.interval_interpolated_drivers = interpolated_drivers[:, int(self.sub_intervals*self.delta_rho_ic):int(self.sub_intervals*drivers.shape[1])]
            self.interval_hourly_drivers = drivers[:, int(time_offset):(drivers.shape[1])]
        else:
            # Fallback to normal propagate_models logic for >= 1 day
            self.propagate_models(init_date, forward_propagation=int(np.ceil(forward_hours/24)))

        if not hasattr(self, 'propagation_drivers'):
            self.t0 = t0
            self.tf = tf
            self.propagation_drivers = interpolated_drivers

        # print(interpolated_drivers)

        f10_value = np.copy(drivers_at_toffset[f10_idx, t0])
        kp_value = np.copy(drivers_at_toffset[kp_idx, t0])
        z_series = self.get_initial_z_from_drivers(self.initial_conditions, f10_value, kp_value)
        z1_k = z_series.values.reshape((self.n_components, 1))
        input_features = ['x_'+ str(k+1).zfill(2) for k in range(z1_k.shape[0])]
        k = 0
        t_span = (0, self.sub_intervals*(tuple(range(drivers.shape[1]))[-1] + 1) - 1) #Start and end times for ivp integration
        t_interval = np.linspace(t_span[0], t_span[1], ((t_span[1] - t_span[0]) + 1)) #time points at which to evaluate the solution
        self.t_interval = t_interval[self.sub_intervals*self.delta_rho_ic:]

        # print(f'Maximum available time T = {(self.sub_intervals*(24*n_days_frcst - hour0)-1)*60}s')
        self.z_results_lst = []

        for chosen_basis_function in list(self.selected_bf_dict.keys()):
            warnings.filterwarnings("ignore")
            ridge_label = 'ridge_parameter_' + "{:.2f}".format(self.selected_bf_dict[chosen_basis_function])

            self.B_sindy_joint_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint']
            self.B_sindy_f10_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['sm_f10']
            self.B_sindy_combined_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['combined']
            self.B_sindy_joint_low_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_low']
            self.B_sindy_joint_mid_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_mid']
            self.B_sindy_joint_high_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_high']


            self.X_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['X_reg_norm_dict_sindy']
            self.Y_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['Y_reg_norm_dict_sindy']
            self.X_library_matrix_inputs_norm_dict = input_data_models['bf_normalization_dict'][chosen_basis_function]['X_library_matrix_inputs_norm_dict']
            
            self.model_params = {'normalization_method': 'std', 'functions': self.basis_functions_dict[chosen_basis_function]}
            self.sindy_tgt_col = self.B_sindy_joint_discrete.shape[1] - n_components + self.pca_coupling
            
            ###########################################sindy###############################################
            array_joint = self.move_column(np.copy(self.B_sindy_joint_discrete), 5, self.sindy_tgt_col)
            A_sindy_joint = np.copy(array_joint[:, -n_components:])
            B_sindy_joint = np.copy(array_joint[:, :(-n_components)])
            A_sindy_joint_c, B_sindy_joint_c = self.discrete_to_continuous(A_sindy_joint, B_sindy_joint, delta_t)

            array_combined = self.move_column(np.copy(self.B_sindy_combined_discrete), 5, self.sindy_tgt_col)
            A_sindy_combined = np.copy(array_combined[:, -n_components:])
            B_sindy_combined = np.copy(array_combined[:, :(-n_components)])
            A_sindy_combined_c, B_sindy_combined_c = self.discrete_to_continuous(A_sindy_combined, B_sindy_combined, delta_t)

            array_f10 = self.move_column(np.copy(self.B_sindy_f10_discrete), 5, self.sindy_tgt_col)
            A_sindy_f10 = np.copy(array_f10[:, -n_components:])
            B_sindy_f10 = np.copy(array_f10[:, :(-n_components)])
            A_sindy_f10_c, B_sindy_f10_c = self.discrete_to_continuous(A_sindy_f10, B_sindy_f10, delta_t)   

            array_joint_low = self.move_column(np.copy(self.B_sindy_joint_low_d), 5, self.sindy_tgt_col)
            A_sindy_joint_low = np.copy(array_joint_low[:, -n_components:])
            B_sindy_joint_low = np.copy(array_joint_low[:, :(-n_components)])
            A_sindy_joint_low_c, B_sindy_joint_low_c = self.discrete_to_continuous(A_sindy_joint_low, B_sindy_joint_low, delta_t)

            array_joint_mid = self.move_column(np.copy(self.B_sindy_joint_mid_d), 5, self.sindy_tgt_col)
            A_sindy_joint_mid = np.copy(array_joint_mid[:, -n_components:])
            B_sindy_joint_mid = np.copy(array_joint_mid[:, :(-n_components)])
            A_sindy_joint_mid_c, B_sindy_joint_mid_c = self.discrete_to_continuous(A_sindy_joint_mid, B_sindy_joint_mid, delta_t)

            array_joint_high = self.move_column(np.copy(self.B_sindy_joint_high_d), 5, self.sindy_tgt_col)
            A_sindy_joint_high = np.copy(array_joint_high[:, -n_components:])
            B_sindy_joint_high = np.copy(array_joint_high[:, :(-n_components)])
            A_sindy_joint_high_c, B_sindy_joint_high_c = self.discrete_to_continuous(A_sindy_joint_high, B_sindy_joint_high, delta_t)
            
  
            X_k_norm = self.build_sindy_dyn_frcst_inputs(z1_k, interpolated_drivers, \
                self.X_library_matrix_inputs_norm_dict, self.X_reg_norm_dict_sindy, self.pca_coupling, \
                    kp_idx, f10_idx, self.model_params, normalization_method, input_features, k = int(k))   
            

            
            qF_norm = self.move_column(np.copy(X_k_norm).T, 5, self.sindy_tgt_col).T
            q0_norm_sindy = np.copy(qF_norm[-n_components:])
            solution_sindy = solve_ivp(
                self.ode_func_sindy,
                t_span,
                q0_norm_sindy.flatten(),
                args = (interpolated_drivers, A_sindy_joint_low_c, B_sindy_joint_low_c, A_sindy_joint_mid_c, \
                    B_sindy_joint_mid_c, A_sindy_joint_high_c, B_sindy_joint_high_c, self.sindy_tgt_col, self.pca_coupling, \
                        self.kp_th),
                method = 'RK45',
                t_eval = t_interval
            )
    
            t = self.sub_intervals*solution_sindy.t

            q_sol_sindy = np.full((len(q0_norm_sindy.flatten()), len(t_interval)), np.nan)
            q_sol_sindy[:, :len(solution_sindy.t)] = solution_sindy.y
            z_sindy = np.copy(q_sol_sindy * self.Y_reg_norm_dict_sindy['x_std'] + self.Y_reg_norm_dict_sindy['x_mean'])
            z_sindy = z_sindy[:, int(self.sub_intervals*self.delta_rho_ic):]
            self.z_results_lst.append(z_sindy)
        
        ###########################################dmd###############################################

        A_dmd = np.copy(np.copy(B_nl_dmd_discrete[:, self.time_variables:(-4)]))
        B_dmd = np.copy(np.concatenate([B_nl_dmd_discrete[:, :self.time_variables], B_nl_dmd_discrete[:, (-4):]], axis = 1))

        A_dmd_c, B_dmd_c = self.discrete_to_continuous(A_dmd, B_dmd, delta_t)


        X_k = np.concatenate([z1_k, interpolated_drivers[f10_idx:, k].reshape((-1, 1)), (interpolated_drivers[f10_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1)), 
            (interpolated_drivers[kp_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1))])
        X_k_norm = u.normalize_with_dict(X_k, self.X_reg_norm_dict_nl_dmd, normalization_method)   
        X_k_norm = np.concatenate([interpolated_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
        q0_norm_dmd = np.copy(X_k_norm[self.time_variables:(-4), :])

        solution_dmd = solve_ivp(
            self.ode_func_dmd,
            t_span,
            q0_norm_dmd.flatten(),
            args = (interpolated_drivers, A_dmd_c, B_dmd_c),
            method = 'RK45',
            t_eval = t_interval
        )
        
        t = self.sub_intervals*solution_dmd.t


        q_sol_dmd = np.full((len(q0_norm_dmd.flatten()), len(t_interval)), np.nan)
        q_sol_dmd[:, :len(solution_dmd.t)] = solution_dmd .y
        z_dmd = np.copy(q_sol_dmd * self.Y_reg_norm_dict_nl_dmd['x_std'] + self.Y_reg_norm_dict_nl_dmd['x_mean'])
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


    # def propagate_models(self, init_date, forward_propagation = 5):

    #     '''
    #     Propagation of the density models
        
    #     Inputs:
    #       init_date: datetime64[ns], date representing the initial propagation time
    #       forward_propagation: int representing the number of days the density is propagated since the chosen year-day_of_year
        
    #     Outputs:
    #       No outputs. Saves results within the object itself.
    #     '''
    #     init_date = pd.to_datetime(init_date)
    #     year = init_date.year
    #     day_of_year = init_date.day_of_year #use date 

    #     start_date = init_date
    #     end_date = datetime(year, 1, 1) + timedelta(days=day_of_year + forward_propagation - 1)

    #     # Generate date series with sub_intervals-seconds resolution for n days
    #     self.date_series = pd.date_range(start=start_date, end = end_date, freq = str(self.sub_intervals) + 's')[:(-1)]
    #     self.hourly_date_series = pd.date_range(start=start_date, end = end_date, freq = 'H')[:(-1)]

    #     n_components = self.n_components
    #     normalization_method = 'std'
    #     gamma = 1
    #     delta_t = gamma*self.sub_intervals 
    #     time_offset = self.delta_rho_ic
    #     t0 = day_of_year * 24 - 24 + self.delta_rho_ic # 24 hours of day doy have passed, so you'd start from doy + 1 without subtracting 24
    #     n_days_frcst = forward_propagation
    #     forward_hours = n_days_frcst * 24
    #     tf = t0 + forward_hours + self.delta_rho_ic
    #     f10_idx = 5
    #     kp_idx = 6
    #     input_data_models = self.input_data_sindy['models_coefficients'][()]
    #     B_nl_dmd_discrete = input_data_models['models_dict']['nl-dmd']['plain']['ridge_parameter_1.00']

    #     IC_idx = np.where(self.drivers[0,:] == year)[0]
    #     biased_ic_indices = np.arange(np.min(IC_idx) - self.delta_rho_ic, np.min(IC_idx) + tf)
    #     drivers_IC = np.copy(self.drivers[:, biased_ic_indices])            
    #     drivers = np.copy(drivers_IC[:, (t0):(tf)])

    #     interpolated_drivers = self.interpolate_matrix_rows(drivers, self.sub_intervals)
    #     self.interval_interpolated_drivers = interpolated_drivers
    #     self.interval_hourly_drivers = drivers[:, int(time_offset):(drivers.shape[1])]
    #     if not hasattr(self, 'propagation_drivers'):
    #         self.t0 = t0
    #         self.tf = tf
    #         self.propagation_drivers = interpolated_drivers

    #     f10_value = np.copy(drivers_IC[f10_idx, t0])
    #     kp_value = np.copy(drivers_IC[kp_idx, t0])
    #     z_series = self.get_initial_z_from_drivers(self.initial_conditions, f10_value, kp_value)
    #     z1_k = z_series.values.reshape((n_components, 1))
    #     input_features = ['x_'+ str(k+1).zfill(2) for k in range(z1_k.shape[0])]
    #     k = 0
    #     t_span = (0, self.sub_intervals*(tuple(range(drivers.shape[1]))[-1] + 1) - 1)
    #     t_interval = np.linspace(t_span[0], t_span[1], ((t_span[1] - t_span[0]) + 1))
    #     self.t_interval = t_interval[self.sub_intervals*self.delta_rho_ic:]

    #     print(f'Maximum available time T = {(self.sub_intervals*24*n_days_frcst-1)*60}s')
    #     self.z_results_lst = []
    #     for chosen_basis_function in list(self.selected_bf_dict.keys()):
    #         warnings.filterwarnings("ignore")
    #         ridge_label = 'ridge_parameter_' + "{:.2f}".format(self.selected_bf_dict[chosen_basis_function])

    #         self.B_sindy_joint_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint']
    #         self.B_sindy_f10_discrete = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['sm_f10']
    #         self.B_sindy_joint_low_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_low']
    #         self.B_sindy_joint_mid_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_mid']
    #         self.B_sindy_joint_high_d = input_data_models['models_dict']['sindy'][chosen_basis_function][ridge_label]['joint_high']
    #         self.X_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['X_reg_norm_dict_sindy']
    #         self.Y_reg_norm_dict_sindy = input_data_models['bf_normalization_dict'][chosen_basis_function]['Y_reg_norm_dict_sindy']
    #         self.X_library_matrix_inputs_norm_dict = input_data_models['bf_normalization_dict'][chosen_basis_function]['X_library_matrix_inputs_norm_dict']
    #         self.model_params = {'normalization_method': 'std', 'functions': self.basis_functions_dict[chosen_basis_function]}
    #         self.sindy_tgt_col = self.B_sindy_joint_discrete.shape[1] - n_components + self.pca_coupling
            
    #         ###########################################sindy###############################################
    #         array_joint = self.move_column(np.copy(self.B_sindy_joint_discrete), 5, self.sindy_tgt_col)
    #         A_sindy_joint = np.copy(array_joint[:, -n_components:])
    #         B_sindy_joint = np.copy(array_joint[:, :(-n_components)])

    #         A_sindy_joint_c, B_sindy_joint_c = self.discrete_to_continuous(A_sindy_joint, B_sindy_joint, delta_t)

    #         array_f10 = self.move_column(np.copy(self.B_sindy_f10_discrete), 5, self.sindy_tgt_col)
    #         A_sindy_f10 = np.copy(array_f10[:, -n_components:])
    #         B_sindy_f10 = np.copy(array_f10[:, :(-n_components)])

    #         array_joint_low = self.move_column(np.copy(self.B_sindy_joint_low_d), 5, self.sindy_tgt_col)
    #         A_sindy_joint_low = np.copy(array_joint_low[:, -n_components:])
    #         B_sindy_joint_low = np.copy(array_joint_low[:, :(-n_components)])
    #         A_sindy_joint_low_c, B_sindy_joint_low_c = self.discrete_to_continuous(A_sindy_joint_low, B_sindy_joint_low, delta_t)

    #         array_joint_mid = self.move_column(np.copy(self.B_sindy_joint_mid_d), 5, self.sindy_tgt_col)
    #         A_sindy_joint_mid = np.copy(array_joint_mid[:, -n_components:])
    #         B_sindy_joint_mid = np.copy(array_joint_mid[:, :(-n_components)])
    #         A_sindy_joint_mid_c, B_sindy_joint_mid_c = self.discrete_to_continuous(A_sindy_joint_mid, B_sindy_joint_mid, delta_t)

    #         array_joint_high = self.move_column(np.copy(self.B_sindy_joint_high_d), 5, self.sindy_tgt_col)
    #         A_sindy_joint_high = np.copy(array_joint_high[:, -n_components:])
    #         B_sindy_joint_high = np.copy(array_joint_high[:, :(-n_components)])
    #         A_sindy_joint_high_c, B_sindy_joint_high_c = self.discrete_to_continuous(A_sindy_joint_high, B_sindy_joint_high, delta_t)

    #         A_sindy_f10_c, B_sindy_f10_c = self.discrete_to_continuous(A_sindy_f10, B_sindy_f10, delta_t)   
    #         X_k_norm = self.build_sindy_dyn_frcst_inputs(z1_k, interpolated_drivers, self.X_library_matrix_inputs_norm_dict, \
    #             self.X_reg_norm_dict_sindy, self.pca_coupling, kp_idx, f10_idx, self.model_params,\
    #             normalization_method, input_features, k = int(k)) 

    #         # X_k_norm = self.build_sindy_dyn_frcst_inputs(z1_k, interpolated_drivers, \
    #         #     self.X_library_matrix_inputs_norm_dict, self.X_reg_norm_dict_sindy, self.pca_coupling, \
    #         #         kp_idx, f10_idx, self.model_params, normalization_method, input_features, k = int(k))  

    #         qF_norm = self.move_column(np.copy(X_k_norm).T, 5, self.sindy_tgt_col).T
    #         q0_norm_sindy = np.copy(qF_norm[-n_components:])
    #         solution_sindy = solve_ivp(
    #             self.ode_func_sindy,
    #             t_span,
    #             q0_norm_sindy.flatten(),
    #             args = (interpolated_drivers, A_sindy_joint_c, B_sindy_joint_c, A_sindy_joint_high_c, \
    #                             B_sindy_joint_high_c, self.sindy_tgt_col, self.pca_coupling, \
    #                     self.kp_th),
    #             method = 'RK45',
    #             t_eval = t_interval
    #         )
    
    #         t = self.sub_intervals*solution_sindy.t

    #         q_sol_sindy = np.full((len(q0_norm_sindy.flatten()), len(t_interval)), np.nan)
    #         q_sol_sindy[:, :len(solution_sindy.t)] = solution_sindy.y
    #         z_sindy = np.copy(q_sol_sindy * self.Y_reg_norm_dict_sindy['x_std'] + self.Y_reg_norm_dict_sindy['x_mean'])
    #         z_sindy = z_sindy[:, int(self.sub_intervals*self.delta_rho_ic):]
    #         self.z_results_lst.append(z_sindy)
        
    #     ###########################################dmd###############################################

    #     A_dmd = np.copy(np.copy(B_nl_dmd_discrete[:, 4:(-4)]))
    #     B_dmd = np.copy(np.concatenate([B_nl_dmd_discrete[:, :4], B_nl_dmd_discrete[:, (-4):]], axis = 1))

    #     A_dmd_c, B_dmd_c = self.discrete_to_continuous(A_dmd, B_dmd, delta_t)


    #     X_k = np.concatenate([z1_k, interpolated_drivers[f10_idx:, k].reshape((-1, 1)), (interpolated_drivers[f10_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1)), 
    #         (interpolated_drivers[kp_idx, k] * interpolated_drivers[kp_idx, k]).reshape((-1, 1))])
    #     X_k_norm = u.normalize_with_dict(X_k, self.X_reg_norm_dict_nl_dmd, normalization_method)   
    #     X_k_norm = np.concatenate([interpolated_drivers[1:f10_idx, k].reshape((-1, 1)), X_k_norm])
    #     q0_norm_dmd = np.copy(X_k_norm[4:(-4), :])

    #     solution_dmd = solve_ivp(
    #         self.ode_func_dmd,
    #         t_span,
    #         q0_norm_dmd.flatten(),
    #         args = (interpolated_drivers, A_dmd_c, B_dmd_c),
    #         method = 'RK45',
    #         t_eval = t_interval
    #     )
        
    #     t = self.sub_intervals*solution_dmd.t


    #     q_sol_dmd = np.full((len(q0_norm_dmd.flatten()), len(t_interval)), np.nan)
    #     q_sol_dmd[:, :len(solution_dmd.t)] = solution_dmd .y
    #     z_dmd = np.copy(q_sol_dmd * self.Y_reg_norm_dict_nl_dmd['x_std'] + self.Y_reg_norm_dict_nl_dmd['x_mean'])
    #     z_dmd = z_dmd[:, self.sub_intervals*self.delta_rho_ic:]
    #     self.z_results_lst.append(z_dmd)
    #     if self.delta_rho_ic != 0:
    #         self.t = t[:(-int(self.sub_intervals*self.delta_rho_ic))]
    #     else:
    #         self.t = t
    #     self.z_dict = {}
    #     models = list(self.selected_bf_dict.keys()) + ['dmd']
    #     for k, z in enumerate(self.z_results_lst):
    #         self.z_dict[models[k]] = self.z_results_lst[k]



class rope_data_interpolator( PythonAtmosphere ):

    def __init__( self, data: rope_propagator, earth_shape: BodyShape = None, sigma_point_value: float = 0.0, lt_low = 0, lt_high = 23.6667, lat_low = -87.5, lat_high = 87.5, alt_low = 100, alt_high = 980 ):
        super().__init__()

        self.data = data
        self.sigma_point_value = sigma_point_value
        self.j2000: Frame = FramesFactory.getEME2000()
        self.utc: UTCScale = TimeScalesFactory.getUTC()
        self.lt_low = lt_low
        self.lt_high = lt_high # 23
        self.lat_low = lat_low # -87.5   
        self.lat_high = lat_high # 87.5
        self.alt_low = alt_low
        self.alt_high = alt_high
        
        # if earth_shape is not None:
        #     self.shape = earth_shape
        # else:
        #     self.shape: BodyShape = OneAxisEllipsoid( Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING, FramesFactory.getITRF( IERSConventions.IERS_2010, False ) )


    def __compute_density__( self, lla: np.array, T: float ) -> tuple[float, float]:
        warnings.filterwarnings("ignore")
        # try:
        # Read command-line arguments
        lla = lla.astype( float )
        T = float( T )  # Delta t in seconds
        point = np.array( [ lla[1], lla[0], lla[2] ] )  # (LAT, LON, ALT) ---> (LON, LAT, ALT)

        # Prepare interpolation variables
        t = self.data.t

        # Define grid for interpolation
        alt0 = 100  # Initial altitude in km
        step = 20.  # Altitude step in km

        lt = np.linspace(self.lt_low, self.lt_high, self.data.U0.shape[0])
        lat = np.linspace( self.lat_low, self.lat_high, self.data.U0.shape[1]) 
        alt = np.arange( alt0, alt0 + step * ( self.data.U0.shape[2] ), step )
        print(alt.shape)
        
        my_interpolating_U0 = rgi( (lt, lat, alt), self.data.U0, bounds_error=False, fill_value=None)
        my_interpolating_mu0 = rgi( (lt, lat, alt), self.data.mu0, bounds_error=False, fill_value=None) 
        
        # Perform interpolation
        U0_interp = my_interpolating_U0(point)
        mu0_interp = my_interpolating_mu0(point)

        sindy_interpolators_lst = []
        for sindy_model in list(self.data.z_dict.keys()):
            sindy_interpolators_lst.append(interp1d(t.flatten(), self.data.z_dict[sindy_model], kind='linear', axis=1, fill_value = "interpolate"))
        
        z_uncertainties_values_lst = [ 10 ** (U0_interp @ interpolator(T) + mu0_interp.reshape((-1, 1))) for interpolator in sindy_interpolators_lst]
        density_std = np.nanstd(z_uncertainties_values_lst)
        density_mean = np.nanmean(z_uncertainties_values_lst)
        
        return np.array([float(item.item()) for item in z_uncertainties_values_lst] + [float(density_mean.item())] + [float(density_std.item())])
    
    
    def interpolate_density_multi_rows( self, Tlla: np.array) -> tuple[float, float]:
        '''
        Function to inyterpolate density using multiple inputs
        
        Inputs:
          Tlla: np.array of shape (n, 3): T (sec) / lon (deg) / lat (deg) / alt (km)
          where n is the number of required inputs
        
        Outputs:
          tuple of atmospheric density output in kg/m^3 and uncertainty variance
        '''
        warnings.filterwarnings("ignore")
        # print(Tlla)
        # try:
        # Read command-line arguments
        lla = Tlla[:, 1:]
        T = Tlla[:, 0]
        # print(T)
        points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])  # (LAT, LON, ALT) ---> (LON, LAT, ALT)

        # Prepare interpolation variables
        t = self.data.t

        # Define grid for interpolation
        alt0 = 100  # Initial altitude in km
        step = 20.  # Altitude step in km

        lt = np.linspace(self.lt_low, self.lt_high, self.data.U0.shape[0])
        # lt = 24. * (np.linspace(self.lt_low, self.lt_high, self.data.U0.shape[0]) - self.lt_low) / (self.lt_high - self.lt_low)
        lat = np.linspace( self.lat_low, self.lat_high, self.data.U0.shape[1]) 
        alt = np.arange( alt0, alt0 + step * ( self.data.U0.shape[2] ), step )
                
        my_interpolating_U0 = rgi( (lt, lat, alt), self.data.U0, bounds_error=False, fill_value=None)
        my_interpolating_mu0 = rgi( (lt, lat, alt), self.data.mu0, bounds_error=False, fill_value=None) 
        
        # Perform interpolation
        U0_interp = my_interpolating_U0(points)
        mu0_interp = my_interpolating_mu0(points)

        sindy_interpolators_lst = []
        for sindy_model in list(self.data.z_dict.keys()):
            sindy_interpolators_lst.append(interp1d(t.flatten(), self.data.z_dict[sindy_model], kind='linear', axis=1, fill_value = "interpolate"))
        # print(T[-1], t.min(), t.max())
        z_uncertainties_values_lst = [ 10 ** (np.sum(U0_interp * interpolator(T).T, axis = 1).reshape((-1, 1)) + mu0_interp.T.reshape((-1, 1))) for interpolator in sindy_interpolators_lst]
        interpolated_models = np.stack(z_uncertainties_values_lst).squeeze(-1).T

        density_std = np.nanstd(interpolated_models, axis = 1)
        density = np.nanmean(interpolated_models, axis = 1)
        # density_poly = interpolated_models[:, 0]
        # density_poly_all = interpolated_models[:, 2]
        density_dmd = interpolated_models[:, -1]

        return interpolated_models, density_dmd, density, density_std

    def interpolate( self, timestamps: np.array, lla: np.array) -> tuple[float, float]:
        '''
        Function to inyterpolate density using multiple inputs
        
        Inputs:
          timestamps: np.array of shape (n, 1): datetime64[ns]
          lla: np.array of shape (n, 3): T (sec) / LST (hour) / lat (deg) / alt (km)
          where n is the number of required inputs
        
        Outputs:
          tuple of atmospheric density output in kg/m^3 and uncertainty variance
        '''
        warnings.filterwarnings("ignore")
        timestamps = pd.to_datetime(timestamps)
        if (self.data.date_series is None) | ~((np.all(timestamps >= self.data.date_series[0])) & (np.all(timestamps <= self.data.date_series[-1]))):    
            if np.ndim(timestamps) == 0 or (hasattr(timestamps, 'shape') and timestamps.shape == ()):
                dates = pd.to_datetime(timestamps)
                prop_date = self.data.date_series[0]
                adjusted_forward_propagation = (dates.max() - prop_date).days + 1
                print(f'System is propagating from {timestamps} for {adjusted_forward_propagation} days')   
                self.data.propagate_models(pd.to_datetime(prop_date), forward_propagation = adjusted_forward_propagation)

            else:
                dates = pd.to_datetime(timestamps)
                prop_date = self.data.date_series[0]
                adjusted_forward_propagation = (dates.max() - prop_date).days + 1
                print(f'System is propagating from {timestamps[0]} for {adjusted_forward_propagation} days')   
                self.data.propagate_models(pd.to_datetime(prop_date), forward_propagation = adjusted_forward_propagation)

        t = self.data.t
        
        T = (timestamps - self.data.date_series[0]).total_seconds()
        
        points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])  # (LAT, LT, ALT) ---> (LT, LAT, ALT)

        alt0 = self.alt_low  # Initial altitude in km
        step = 20.  # Altitude step in km

        lt = np.linspace(self.lt_low, self.lt_high, self.data.U0.shape[0])
        lat = np.linspace( self.lat_low, self.lat_high, self.data.U0.shape[1]) 
        alt = np.linspace(self.alt_low, self.alt_high, self.data.U0.shape[2])
        
        my_interpolating_U0 = rgi( (lt, lat, alt), self.data.U0, bounds_error=False, fill_value=None)
        my_interpolating_mu0 = rgi( (lt, lat, alt), self.data.mu0, bounds_error=False, fill_value=None) 
        
        # Perform interpolation
        U0_interp = my_interpolating_U0(points).reshape((-1, 10))
        mu0_interp = my_interpolating_mu0(points).reshape((-1, 1))

        sindy_interpolators_lst = []
        for sindy_model in list(self.data.z_dict.keys()):
            sindy_interpolators_lst.append(interp1d(t.flatten(), self.data.z_dict[sindy_model], \
                kind='linear', axis=1, bounds_error=False, fill_value = None))

        # density_models_values_lst = []
        # for interpolator in sindy_interpolators_lst:
        #     rho_model = 10. ** ( np.sum(U0_interp * interpolator(T).T, axis = 1).reshape((-1, 1)) + \
        #     mu0_interp.T.reshape((-1, 1)) )
        #     density_models_values_lst.append(rho_model)
        
        density_models_values_lst = [ 10. ** ( np.sum(U0_interp * interpolator(T).T, axis = 1).reshape((-1, 1)) + \
            mu0_interp.T.reshape((-1, 1)) ) for interpolator in sindy_interpolators_lst]
        
        interpolated_models = np.stack(density_models_values_lst).squeeze(-1).T

        density_std = np.nanstd(interpolated_models[:, :], axis = 1)
        density = np.nanmean(interpolated_models[:, :-1], axis = 1)
        # density_poly = interpolated_models[:, 0]
        # density_poly_all = interpolated_models[:, 2]
        density_dmd = interpolated_models[:, -1]

        return interpolated_models, density_dmd, density, density_std

    def interpolate_full_grid( self, timestamps: np.array, lla: np.array, forward_propagation: int = 3) -> tuple[float, float]:
        warnings.filterwarnings("ignore")
        timestamps = pd.to_datetime(timestamps)
        if (self.data.date_series is None) | ~((np.all(timestamps >= self.data.date_series[0])) & (np.all(timestamps <= self.data.date_series[-1]))):    
            if np.ndim(timestamps) == 0 or (hasattr(timestamps, 'shape') and timestamps.shape == ()):
                dates = pd.to_datetime(timestamps)
                adjusted_forward_propagation = (dates.max() - dates.min()).days + 1
                print(f'System is propagating from {timestamps} for {adjusted_forward_propagation} days')   
                self.data.propagate_models(pd.to_datetime(timestamps), forward_propagation = adjusted_forward_propagation)

            else:
                dates = pd.to_datetime(timestamps)
                adjusted_forward_propagation = (dates.max() - dates.min()).days + 1
                print(f'System is propagating from {timestamps[0]} for {adjusted_forward_propagation} days')   
                self.data.propagate_models(pd.to_datetime(timestamps[0]), forward_propagation = adjusted_forward_propagation)

        t = self.data.t
        
        T = (timestamps - self.data.date_series[0]).total_seconds()
        
        points = np.column_stack([lla[:, 1], lla[:, 0], lla[:, 2]])  # Interpolation points, (LAT, LT, ALT) ---> (LT, LAT, ALT)


        lt = np.linspace(self.lt_low, self.lt_high, self.data.U0.shape[0])
        lat = np.linspace( self.lat_low, self.lat_high, self.data.U0.shape[1]) 
        alt = np.linspace(self.alt_low, self.alt_high, self.data.U0.shape[2])
        
        my_interpolating_U0 = rgi( (lt, lat, alt), self.data.U0, bounds_error=False, fill_value=None)
        my_interpolating_mu0 = rgi( (lt, lat, alt), self.data.mu0, bounds_error=False, fill_value=None) 

        sindy_interpolators_lst = []
        for sindy_model in list(self.data.z_dict.keys()):
            sindy_interpolators_lst.append(interp1d(t.flatten(), self.data.z_dict[sindy_model], \
                kind='linear', axis=1, bounds_error=False, fill_value = None))

        by_interpolator_lst = []

        for interpolator in sindy_interpolators_lst:
            interpolated_vals = interpolator(T)  
            logrho = self.data.U0 @ interpolated_vals + self.data.mu0[:, :, :, None]
            by_time_lst = []

            for i in range(logrho.shape[3]):
                my_interpolating_logrho = rgi(
                    (lt, lat, alt), logrho[:, :, :, i],
                    bounds_error=False, fill_value=None
                )
                by_time_lst.append(10 ** my_interpolating_logrho(points[i]))

            stacked_models = np.array(by_time_lst).reshape((1, -1))
            by_interpolator_lst.append(stacked_models)

        density_models_values = np.concatenate(by_interpolator_lst, axis=0).T


        density_std = np.nanstd(density_models_values[:, :-1], axis = 1)
        density = np.nanmean(density_models_values[:, :-1], axis = 1)
        density_dmd = density_models_values[:, -1]

        return density_models_values, density_dmd, density, density_std


def run(initial_propagation_date_str : str, interpolation_dates: np.ndarray, lla_array : np.ndarray, drivers : str = 'sw_all_years_preprocessed.csv', forward_propagation : int = 3):
    
    sw_all_years = pd.read_csv(f'./sw_inputs/{drivers}', sep = ',').drop(columns = ['datetime'])
    sw_drivers_all_years = np.copy(sw_all_years.values.T)

    sindy = rope_propagator(drivers = sw_drivers_all_years)
    sindy.propagate_models(init_date = initial_propagation_date_str, forward_propagation = forward_propagation)
    rope_density = rope_data_interpolator( data = sindy )

    _, _, density, density_std = rope_density.interpolate(interpolation_dates, lla_array)

    return density, density_std